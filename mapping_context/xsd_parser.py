import xmlschema
from typing import List, Dict

def xsd_to_flat_schema(xsd_content: str) -> List[Dict]:
    # Parse the XSD content
    schema = xmlschema.XMLSchema(xsd_content)
    elements = []

    # Iterate through all global elements (like MT_Empdetails_01)
    for root_name, root_elem in schema.elements.items():
        # iter() traverses the entire tree of the element
        for xsd_element in root_elem.iter():
            if hasattr(xsd_element, 'name') and xsd_element.name:
                # Handle 'unbounded' cardinality correctly
                max_occ = xsd_element.max_occurs
                if max_occ is None:
                    max_occ = "unbounded"
                
                # Determine type name or label as complexType
                if hasattr(xsd_element.type, 'name') and xsd_element.type.name:
                    type_info = xsd_element.type.name
                else:
                    type_info = "complexType"

                elements.append({
                    "path": xsd_element.get_path(), # Gets the full XPath
                    "type": type_info,
                    "required": xsd_element.min_occurs > 0,
                    "max_occurs": max_occ
                })

    return elements