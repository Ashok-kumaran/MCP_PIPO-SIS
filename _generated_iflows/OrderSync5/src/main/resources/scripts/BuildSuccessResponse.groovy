import com.sap.gateway.ip.core.customdev.util.Message
import groovy.json.JsonOutput

Message processData(Message message) {
    def body = message.getBody(String)
    String salesOrder = null
    try {
        def xml = new XmlSlurper().parseText(body)
        salesOrder = (xml.'**'.find { it.name() == 'SalesOrder' }?.text()) ?: null
    } catch (Exception e) {
        // Try JSON
        try {
            def json = new groovy.json.JsonSlurper().parseText(body)
            salesOrder = json?.d?.SalesOrder ?: json?.SalesOrder ?: json?.d?.results?.SalesOrder
        } catch (ignored) {}
    }
    def resp = [status: 'success', salesOrder: salesOrder]
    message.setHeader('Content-Type', 'application/json')
    message.setBody(JsonOutput.toJson(resp))
    return message
}
