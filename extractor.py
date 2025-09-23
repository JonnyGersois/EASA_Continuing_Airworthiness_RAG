# extractor.py

from lxml import etree
from langchain_core.documents import Document

NAMESPACES = {
    "pkg": "http://schemas.microsoft.com/office/2006/xmlPackage",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "er": "http://www.easa.europa.eu/erules-export"
}


def extract_clean_xml_from_package(xml_path: str, save_clean_path: str = None) -> str:
    """
    Extract body (/word/document.xml) and metadata (/customXml/itemX.xml) from the Office xmlPackage.
    Returns a clean XML string and optionally saves to file.
    """
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    # Body (WordprocessingML)
    word_part = root.xpath(
        '//pkg:part[@pkg:name="/word/document.xml"]/pkg:xmlData/w:document',
        namespaces=NAMESPACES)
    if not word_part:
        raise ValueError("❌ Could not find /word/document.xml")
    word_doc = word_part[0]

    # Metadata (EASA er:document)
    er_part = root.xpath(
        '//pkg:part[contains(@pkg:name,"/customXml/item")]/pkg:xmlData/er:document',
        namespaces=NAMESPACES)
    if not er_part:
        raise ValueError("❌ Could not find er:document in /customXml/")
    er_doc = er_part[0]

    # Merge into a simple container
    clean_root = etree.Element("EASA_Clean")
    clean_root.append(word_doc)
    clean_root.append(er_doc)

    clean_str = etree.tostring(clean_root, pretty_print=True, encoding="unicode")

    if save_clean_path:
        with open(save_clean_path, "w", encoding="utf-8") as f:
            f.write(clean_str)

    return clean_str


def convert_xml_to_documents(clean_xml_str: str, window: int = 0) -> list[Document]:
    """
    Convert cleaned XML into LangChain Documents: 
    each <er:topic> matched with its <w:sdt> text and metadata.
    """
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    tree = etree.fromstring(clean_xml_str.encode("utf-8"), parser=parser)

    docs = []

    # Map sdt-id -> text
    sdt_map = {}
    for sdt in tree.xpath(".//w:sdt", namespaces=NAMESPACES):
        w_id = sdt.find(".//w:sdtPr/w:id", namespaces=NAMESPACES)
        if w_id is None:
            continue
        sdt_key = w_id.attrib.get(f"{{{NAMESPACES['w']}}}val")
        texts = [t.text for t in sdt.findall(".//w:t", namespaces=NAMESPACES) if t.text]
        if texts:
            sdt_map[sdt_key] = " ".join(texts)

    # Attach metadata from <er:topic>
    for topic in tree.xpath(".//er:topic", namespaces=NAMESPACES):
        sdt_id = topic.attrib.get("sdt-id")
        body_text = sdt_map.get(sdt_id, "")
        if not body_text:
            continue

        metadata = {k: v for k, v in topic.attrib.items()}
        title_el = topic.find(".//er:title", namespaces=NAMESPACES)
        if title_el is not None and title_el.text:
            metadata["title"] = title_el.text.strip()

        content = (metadata.get("title", "") + "\n\n" + body_text).strip()
        docs.append(Document(page_content=content, metadata=metadata))

    return docs