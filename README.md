# PIPO to SAP Integration Suite 

## Using Python client and ts SERVER: mcp-integration-suite

## Instructions
1. Used Python client to connect ts based mcp server
2. Run : uv run <python client (dot)py > <mcp-server /dist/index (dot) js>


App:
- python client(dot)py
- js mcp server

mcp server link: https://github.com/1nbuc/mcp-integration-suite

mcp server blog: https://community.sap.com/t5/technology-blog-posts-by-members/using-integration-suite-with-the-power-of-ai/ba-p/14067293


pipo_project/
├─ pipo_client.py
├─ graph/
│  ├─ planner.py
│  ├─ executor.py
│  ├─ state.py
│  └─ tools_node.py
├─ mcp_utils/
│  ├─ callbacks.py
│  ├─ schema_parser.py
│  └─ file_manager.py
└─ prompts/
   ├─ planner_prompt.txt
   └─ executor_prompt.txt
