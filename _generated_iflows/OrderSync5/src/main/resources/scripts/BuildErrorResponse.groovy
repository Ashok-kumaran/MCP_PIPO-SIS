import com.sap.gateway.ip.core.customdev.util.Message
import groovy.json.JsonOutput

Message processData(Message message) {
    def errMsg = message.getProperty('SAP_ErrorMessage') ?: message.getBody(String)
    def resp = [status: 'error', message: String.valueOf(errMsg)]
    message.setHeader('Content-Type', 'application/json')
    message.setHeader('CamelHttpResponseCode', 500)
    message.setBody(JsonOutput.toJson(resp))
    return message
}
