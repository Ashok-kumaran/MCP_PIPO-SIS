import asyncio
import sys
import os
import json
import logging
from typing import Optional, Any, Type
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import create_model, BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool


# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    filename="pipo_client.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


# ==========================================================
# BPMN TEMPLATE GENERATORS
# ==========================================================
def generate_sftp_to_rest_bpmn(iflow_id: str) -> str:
    """Generate BPMN for SFTP → REST API scenario"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn2:definitions xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL"
                   xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
                   xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"
                   xmlns:di="http://www.omg.org/spec/DD/20100524/DI"
                   xmlns:ifl="http://sap.com/xi/ifl"
                   id="{iflow_id}_Definitions">
  <bpmn2:collaboration id="{iflow_id}_Collaboration">
    <bpmn2:participant id="Participant_SFTP" name="SFTP Sender" processRef="{iflow_id}_Process"/>
  </bpmn2:collaboration>
  
  <bpmn2:process id="{iflow_id}_Process" name="{iflow_id}" isExecutable="true">
    <bpmn2:startEvent id="StartEvent_SFTP" name="Start">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>SFTP</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Sender</value>
        </ifl:property>
        <ifl:property>
          <key>address</key>
          <value>/input</value>
        </ifl:property>
        <ifl:property>
          <key>fileName</key>
          <value>*</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:outgoing>SequenceFlow_1</bpmn2:outgoing>
    </bpmn2:startEvent>
    
    <bpmn2:endEvent id="EndEvent_REST" name="End">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>HTTP</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Receiver</value>
        </ifl:property>
        <ifl:property>
          <key>httpMethod</key>
          <value>POST</value>
        </ifl:property>
        <ifl:property>
          <key>address</key>
          <value>/api/upload</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>SequenceFlow_1</bpmn2:incoming>
    </bpmn2:endEvent>
    
    <bpmn2:sequenceFlow id="SequenceFlow_1" sourceRef="StartEvent_SFTP" targetRef="EndEvent_REST"/>
  </bpmn2:process>
</bpmn2:definitions>"""


def generate_https_json_to_xml_sftp_bpmn(iflow_id: str) -> str:
    """Generate BPMN for HTTPS JSON → XML → SFTP scenario"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn2:definitions xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL"
                   xmlns:ifl="http://sap.com/xi/ifl"
                   id="{iflow_id}_Definitions">
  <bpmn2:process id="{iflow_id}_Process" name="{iflow_id}" isExecutable="true">
    <bpmn2:startEvent id="StartEvent_HTTPS" name="HTTPS Sender">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>HTTPS</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Sender</value>
        </ifl:property>
        <ifl:property>
          <key>address</key>
          <value>/json-to-xml</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:outgoing>Flow_1</bpmn2:outgoing>
    </bpmn2:startEvent>
    
    <bpmn2:serviceTask id="Task_JSONtoXML" name="JSON to XML Converter">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>Converter</value>
        </ifl:property>
        <ifl:property>
          <key>converterType</key>
          <value>JSONToXML</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>Flow_1</bpmn2:incoming>
      <bpmn2:outgoing>Flow_2</bpmn2:outgoing>
    </bpmn2:serviceTask>
    
    <bpmn2:endEvent id="EndEvent_SFTP" name="SFTP Receiver">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>SFTP</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Receiver</value>
        </ifl:property>
        <ifl:property>
          <key>directory</key>
          <value>/output</value>
        </ifl:property>
        <ifl:property>
          <key>fileName</key>
          <value>output.xml</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>Flow_2</bpmn2:incoming>
    </bpmn2:endEvent>
    
    <bpmn2:sequenceFlow id="Flow_1" sourceRef="StartEvent_HTTPS" targetRef="Task_JSONtoXML"/>
    <bpmn2:sequenceFlow id="Flow_2" sourceRef="Task_JSONtoXML" targetRef="EndEvent_SFTP"/>
  </bpmn2:process>
</bpmn2:definitions>"""


def generate_rest_to_jms_bpmn(iflow_id: str) -> str:
    """Generate BPMN for REST → JMS Queue scenario"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn2:definitions xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL"
                   xmlns:ifl="http://sap.com/xi/ifl"
                   id="{iflow_id}_Definitions">
  <bpmn2:process id="{iflow_id}_Process" name="{iflow_id}" isExecutable="true">
    <bpmn2:startEvent id="StartEvent_REST" name="REST Sender">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>HTTPS</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Sender</value>
        </ifl:property>
        <ifl:property>
          <key>address</key>
          <value>/rest-to-jms</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:outgoing>Flow_1</bpmn2:outgoing>
    </bpmn2:startEvent>
    
    <bpmn2:endEvent id="EndEvent_JMS" name="JMS Queue">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>JMS</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Receiver</value>
        </ifl:property>
        <ifl:property>
          <key>queueName</key>
          <value>InputQueue</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>Flow_1</bpmn2:incoming>
    </bpmn2:endEvent>
    
    <bpmn2:sequenceFlow id="Flow_1" sourceRef="StartEvent_REST" targetRef="EndEvent_JMS"/>
  </bpmn2:process>
</bpmn2:definitions>"""


def generate_timer_error_email_bpmn(iflow_id: str) -> str:
    """Generate BPMN for Timer → Error Collection → Email scenario"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn2:definitions xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL"
                   xmlns:ifl="http://sap.com/xi/ifl"
                   id="{iflow_id}_Definitions">
  <bpmn2:process id="{iflow_id}_Process" name="{iflow_id}" isExecutable="true">
    <bpmn2:startEvent id="StartEvent_Timer" name="Timer Start">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>Timer</value>
        </ifl:property>
        <ifl:property>
          <key>schedule</key>
          <value>daily</value>
        </ifl:property>
        <ifl:property>
          <key>time</key>
          <value>00:00:00</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:outgoing>Flow_1</bpmn2:outgoing>
    </bpmn2:startEvent>
    
    <bpmn2:serviceTask id="Task_CollectErrors" name="Collect Error Artifacts">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>Script</value>
        </ifl:property>
        <ifl:property>
          <key>scriptType</key>
          <value>Groovy</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>Flow_1</bpmn2:incoming>
      <bpmn2:outgoing>Flow_2</bpmn2:outgoing>
    </bpmn2:serviceTask>
    
    <bpmn2:endEvent id="EndEvent_Mail" name="Send Email">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>Mail</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Receiver</value>
        </ifl:property>
        <ifl:property>
          <key>subject</key>
          <value>Daily Error Report</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>Flow_2</bpmn2:incoming>
    </bpmn2:endEvent>
    
    <bpmn2:sequenceFlow id="Flow_1" sourceRef="StartEvent_Timer" targetRef="Task_CollectErrors"/>
    <bpmn2:sequenceFlow id="Flow_2" sourceRef="Task_CollectErrors" targetRef="EndEvent_Mail"/>
  </bpmn2:process>
</bpmn2:definitions>"""


def generate_email_to_sftp_bpmn(iflow_id: str) -> str:
    """Generate BPMN for Email → SFTP scenario"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn2:definitions xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL"
                   xmlns:ifl="http://sap.com/xi/ifl"
                   id="{iflow_id}_Definitions">
  <bpmn2:process id="{iflow_id}_Process" name="{iflow_id}" isExecutable="true">
    <bpmn2:startEvent id="StartEvent_Mail" name="Email Sender">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>Mail</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Sender</value>
        </ifl:property>
        <ifl:property>
          <key>processAttachments</key>
          <value>true</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:outgoing>Flow_1</bpmn2:outgoing>
    </bpmn2:startEvent>
    
    <bpmn2:endEvent id="EndEvent_SFTP" name="SFTP Receiver">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>SFTP</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Receiver</value>
        </ifl:property>
        <ifl:property>
          <key>directory</key>
          <value>/attachments</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>Flow_1</bpmn2:incoming>
    </bpmn2:endEvent>
    
    <bpmn2:sequenceFlow id="Flow_1" sourceRef="StartEvent_Mail" targetRef="EndEvent_SFTP"/>
  </bpmn2:process>
</bpmn2:definitions>"""


def generate_https_value_mapping_rest_bpmn(iflow_id: str) -> str:
    """Generate BPMN for HTTPS JSON → Value Mapping → REST scenario"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn2:definitions xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL"
                   xmlns:ifl="http://sap.com/xi/ifl"
                   id="{iflow_id}_Definitions">
  <bpmn2:process id="{iflow_id}_Process" name="{iflow_id}" isExecutable="true">
    <bpmn2:startEvent id="StartEvent_HTTPS" name="HTTPS Sender">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>HTTPS</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Sender</value>
        </ifl:property>
        <ifl:property>
          <key>address</key>
          <value>/json-mapping</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:outgoing>Flow_1</bpmn2:outgoing>
    </bpmn2:startEvent>
    
    <bpmn2:serviceTask id="Task_ValueMapping" name="Value Mapping">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>ValueMapping</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>Flow_1</bpmn2:incoming>
      <bpmn2:outgoing>Flow_2</bpmn2:outgoing>
    </bpmn2:serviceTask>
    
    <bpmn2:endEvent id="EndEvent_REST" name="REST Receiver">
      <bpmn2:extensionElements>
        <ifl:property>
          <key>componentType</key>
          <value>HTTP</value>
        </ifl:property>
        <ifl:property>
          <key>direction</key>
          <value>Receiver</value>
        </ifl:property>
        <ifl:property>
          <key>httpMethod</key>
          <value>POST</value>
        </ifl:property>
        <ifl:property>
          <key>address</key>
          <value>/api/endpoint</value>
        </ifl:property>
      </bpmn2:extensionElements>
      <bpmn2:incoming>Flow_2</bpmn2:incoming>
    </bpmn2:endEvent>
    
    <bpmn2:sequenceFlow id="Flow_1" sourceRef="StartEvent_HTTPS" targetRef="Task_ValueMapping"/>
    <bpmn2:sequenceFlow id="Flow_2" sourceRef="Task_ValueMapping" targetRef="EndEvent_REST"/>
  </bpmn2:process>
</bpmn2:definitions>"""


def select_bpmn_template(query: str) -> tuple[str, str]:
    """
    Select appropriate BPMN template based on query keywords.
    Returns (template_function_name, suggested_iflow_id)
    """
    query_lower = query.lower()
    
    if "sftp" in query_lower and ("rest" in query_lower or "api" in query_lower) and "https" not in query_lower:
        return "sftp_to_rest", "SFTPToREST"
    elif "https" in query_lower and "json" in query_lower and "xml" in query_lower and "sftp" in query_lower:
        return "https_json_xml_sftp", "HTTPSJsonToXMLSFTP"
    elif "rest" in query_lower and "jms" in query_lower:
        return "rest_to_jms", "RESTToJMS"
    elif "timer" in query_lower and "error" in query_lower and "email" in query_lower:
        return "timer_error_email", "DailyErrorReport"
    elif "email" in query_lower and "sftp" in query_lower and "attachment" in query_lower:
        return "email_to_sftp", "EmailAttachmentsToSFTP"
    elif "value mapping" in query_lower or "mapping" in query_lower:
        return "https_value_mapping_rest", "HTTPSValueMappingREST"
    else:
        return "generic", "GenericIFlow"


# ==========================================================
# SAP GEN AI HUB LLM
# ==========================================================
def create_sap_llm():
    deployment_id = os.getenv("LLM_DEPLOYMENT_ID")
    if not deployment_id:
        raise RuntimeError("LLM_DEPLOYMENT_ID missing in .env")

    return ChatOpenAI(
        deployment_id=deployment_id,
        temperature=0,
    )


# ==========================================================
# ASYNC MCP TOOL WRAPPER
# ==========================================================
class MCPAsyncTool(BaseTool):
    """
    LangChain tool that calls an MCP tool asynchronously using the SAME event loop.
    """

    name: str
    description: str
    args_schema: Type[BaseModel]
    session: ClientSession
    mcp_tool_name: str

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Sync run is not supported; use async.")

    async def _arun(self, *args, **kwargs) -> str:
        logger.info(f"[MCP-TOOL] Executing → {self.mcp_tool_name} | Args = {kwargs}")

        result = await self.session.call_tool(self.mcp_tool_name, kwargs)

        if not result.content:
            logger.info(f"[MCP-TOOL] {self.mcp_tool_name} returned EMPTY content")
            return ""

        outputs = []
        for c in result.content:
            if getattr(c, "text", None):
                outputs.append(c.text)
            elif getattr(c, "json", None):
                outputs.append(json.dumps(c.json, ensure_ascii=False))
            else:
                outputs.append(str(c))

        logger.info(f"[MCP-TOOL] Final Output ({self.mcp_tool_name}): {outputs}")

        return "\n".join(outputs)


# ==========================================================
# MCP CLIENT
# ==========================================================
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

        self.llm = create_sap_llm()
        self.agent_tools: list[BaseTool] = []

        self.supervisor: Optional[Any] = None
        self.worker_agent: Optional[AgentExecutor] = None

    # ------------------------------------------------------
    async def connect_to_server(self, server_script: str):

        if server_script.endswith(".py"):
            cmd = "python"
        elif server_script.endswith(".js"):
            cmd = "node"
        else:
            raise ValueError("Server script must be .py or .js")

        params = StdioServerParameters(command=cmd, args=[server_script])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        await self._build_agent_tools()
        await self._build_supervisor()
        await self._build_worker_agent()

        tools = [t.name for t in (await self.session.list_tools()).tools]
        print("\nMCP Connected! Tools:", tools)
        logger.info(f"[MCP] Connected with tools: {tools}")

    # ======================================================
    async def _build_agent_tools(self):

        assert self.session is not None
        tool_list = await self.session.list_tools()

        for tdef in tool_list.tools:
            schema = tdef.inputSchema or {}
            props = schema.get("properties", {}) or {}
            required = set(schema.get("required", []))

            fields: dict[str, tuple[Any, Any]] = {}
            for key in props:
                if key in required:
                    fields[key] = (Any, ...)
                else:
                    fields[key] = (Any, None)

            InputModel = create_model(f"{tdef.name}_Input", **fields)

            tool = MCPAsyncTool(
                name=tdef.name,
                description=tdef.description or "",
                args_schema=InputModel,
                session=self.session,
                mcp_tool_name=tdef.name,
            )

            self.agent_tools.append(tool)

        logger.info(f"[MCP] Loaded {len(self.agent_tools)} tools into LangChain.")

    # ======================================================
    async def _build_supervisor(self):
        """Build supervisor agent for planning"""
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a SUPERVISOR for SAP CPI iFlow creation.\n\n"
             "Your job is to analyze the user's request and create a concise execution plan.\n\n"
             "Identify:\n"
             "1. What type of iFlow is needed (SFTP→REST, HTTPS→XML→SFTP, etc.)\n"
             "2. Package name to use or create\n"
             "3. iFlow ID to create\n"
             "4. Key components needed (adapters, converters, etc.)\n\n"
             "Keep your plan brief and actionable."),
            ("human", "{input}")
        ])
        
        self.supervisor = supervisor_prompt | self.llm | StrOutputParser()
        logger.info("[SUPERVISOR] Supervisor agent created")

    # ======================================================
    async def _build_worker_agent(self):

        worker_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the WORKER agent for SAP CPI iFlow creation.\n\n"
             "CRITICAL RULES:\n"
             "1. NEVER call the same tool twice with identical arguments\n"
             "2. After seeing 'successfully updated' from update-iflow, STOP immediately\n"
             "3. Do NOT keep calling update-iflow repeatedly\n"
             "4. Use the exact BPMN content provided in the context\n"
             "5. Package IDs must be alphanumeric (no special chars except underscore)\n\n"
             "CORRECT update-iflow FORMAT:\n"
             "```json\n"
             "{{\n"
             '  "id": "IFlowID",\n'
             '  "files": [\n'
             "    {{\n"
             '      "filepath": "src/main/resources/scenarioflows/integrationflow/IFlowID.iflw",\n'
             '      "content": "<xml content here>"\n'
             "    }}\n"
             "  ],\n"
             '  "autoDeploy": false\n'
             "}}\n"
             "```\n"
             "IMPORTANT: The 'files' parameter is an ARRAY of objects, each with 'filepath' and 'content' keys.\n\n"
             "EXECUTION WORKFLOW:\n"
             "Step 1: Check if package exists using 'package' tool\n"
             "  - If 404/Not Found, create it with 'create-package'\n"
             "  - If 409/Conflict, package already exists, proceed\n"
             "Step 2: Create empty iFlow with 'create-empty-iflow'\n"
             "Step 3: Update iFlow with provided BPMN using 'update-iflow' (use correct format above)\n"
             "  - Use the BPMN template from context\n"
             "  - Set autoDeploy to false\n"
             "Step 4: Report completion and STOP\n\n"
             "STOPPING CRITERIA:\n"
             "- Once you see status 200 and 'successfully updated', your job is DONE\n"
             "- Do NOT attempt additional updates\n"
             "- Provide a final summary and stop\n\n"
             "Use tools efficiently. Never invent data."),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.agent_tools,
            prompt=worker_prompt,
        )

        self.worker_agent = AgentExecutor(
            agent=agent,
            tools=self.agent_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=12,  # Reduced from 15
            max_execution_time=300,  # 5 minute timeout
            early_stopping_method="generate"  # Generate answer when stopping
        )

    # ======================================================
    async def process_query(self, query: str):

        logger.info(f"[USER] Query: {query}")

        # Step 1: Supervisor creates plan
        print("\n[SUPERVISOR] Analyzing request and creating plan...")
        plan = await self.supervisor.ainvoke({"input": query})
        logger.info(f"[SUPERVISOR] Plan:\n{plan}")
        print(f"\n[PLAN]\n{plan}\n")

        # Step 2: Detect scenario and prepare BPMN template
        template_type, suggested_id = select_bpmn_template(query)
        logger.info(f"[TEMPLATE] Selected: {template_type}, Suggested ID: {suggested_id}")
        
        # Generate BPMN content based on template type
        bpmn_generators = {
            "sftp_to_rest": generate_sftp_to_rest_bpmn,
            "https_json_xml_sftp": generate_https_json_to_xml_sftp_bpmn,
            "rest_to_jms": generate_rest_to_jms_bpmn,
            "timer_error_email": generate_timer_error_email_bpmn,
            "email_to_sftp": generate_email_to_sftp_bpmn,
            "https_value_mapping_rest": generate_https_value_mapping_rest_bpmn,
        }
        
        bpmn_content = None
        if template_type in bpmn_generators:
            bpmn_content = bpmn_generators[template_type](suggested_id)
            logger.info(f"[BPMN] Generated template for {template_type}")

        # Step 3: Execute with worker agent
        print("\n[WORKER] Executing iFlow creation...")
        
        enhanced_query = f"""
{query}

EXECUTION CONTEXT:
- Suggested Package ID: {suggested_id}Pkg (alphanumeric, no special chars)
- Suggested iFlow ID: {suggested_id}
- Template Type: {template_type}

BPMN TEMPLATE TO USE:
```xml
{bpmn_content if bpmn_content else "Use basic BPMN structure"}
```

FORMAT FOR update-iflow:
- files must be an ARRAY: [{{ "filepath": "...", "content": "..." }}]
- filepath must be: "src/main/resources/scenarioflows/integrationflow/{suggested_id}.iflw"
- content must be the complete BPMN XML string

INSTRUCTIONS:
1. First check if package exists
2. Create package if needed (handle 409 Conflict gracefully)
3. Create empty iFlow
4. Update iFlow with the BPMN template above using correct format
5. Stop after successful update

Remember: After you see "successfully updated", STOP immediately.
"""

        if not self.worker_agent:
            raise RuntimeError("Worker agent not initialized")

        try:
            worker_out = await self.worker_agent.ainvoke(
                {"input": enhanced_query},
                return_intermediate_steps=True
            )

            answer = worker_out.get("output", "Task execution completed")
            steps = worker_out.get("intermediate_steps", [])
            # Check execution status
            steps_str = str(steps)
            if "successfully updated" in steps_str.lower():
                logger.info("[SUCCESS] ✓ iFlow created successfully")
                answer = f"✓ SUCCESS: iFlow '{suggested_id}' created successfully in package '{suggested_id}Pkg'.\n\n{answer}"
            elif "max iterations" in answer.lower() or len(steps) >= 12:
                logger.warning("[WARNING] Agent hit iteration limit")
                answer = f"⚠ PARTIAL SUCCESS: iFlow creation started but hit iteration limit. Please check the CPI tenant.\n\n{answer}"
            
            logger.info(f"[FINAL] Answer: {answer}")
            return answer

        except Exception as e:
            logger.exception("Error in worker execution")
            return f"❌ ERROR: {str(e)}"

    # ======================================================
    async def chat_loop(self):
        print("\n" + "="*60)
        print("SAP CPI iFlow Generator - Ready!")
        print("="*60)
        print("\nExample queries:")
        print("  1. Create a CPI iFlow with SFTP sender and REST receiver")
        print("  2. Create iFlow: HTTPS JSON → XML → SFTP")
        print("  3. Create REST to JMS iFlow")
        print("  4. Create daily timer iFlow to collect errors and send email")
        print("  5. Create iFlow to save email attachments to SFTP")
        print("  6. Create HTTPS to REST iFlow with value mapping")
        print("\nType 'quit' to exit.\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n💬 Query: ")
                if user_input.lower() in ("quit", "exit", "q"):
                    print("\nGoodbye! 👋")
                    break

                if not user_input.strip():
                    continue

                answer = await self.process_query(user_input)
                print("\n" + "="*60)
                print("RESPONSE:")
                print("="*60)
                print(answer)
                print("="*60)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                logger.exception("Chat loop error")

    # ======================================================
    async def cleanup(self):
        await self.exit_stack.aclose()
        logger.info("[CLEANUP] MCP session closed")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python pipo_client.py <server.js|server.py>")
        print("\nExample:")
        print("  python pipo_client.py server.js")
        sys.exit(1)

    server = sys.argv[1]
    client = MCPClient()

    try:
        await client.connect_to_server(server)
        await client.chat_loop()
    except Exception as e:
        logger.exception("Fatal error in main")
        print(f"\n❌ FATAL ERROR: {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown requested. Goodbye! 👋")
