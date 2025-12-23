from mapping_context.xsd_context_builder import build_schema_context

if __name__ == "__main__":
    test_query = """
    Create message mapping using:
    Source XSD: s3://hcp-9590b8a4-fcd8-4832-a678-cc655208784e/demo/XSLT source.xsd
    Target XSD: s3://hcp-9590b8a4-fcd8-4832-a678-cc655208784e/demo/XSLT target.xsd
    """

    context = build_schema_context(test_query)

    print("\n======= FINAL SCHEMA CONTEXT =======\n")
    print(context)
