import com.sap.gateway.ip.core.customdev.util.Message
import groovy.json.JsonSlurper

Message processData(Message message) {
    def body = message.getBody(String) as String
    def json = new JsonSlurper().parseText(body)

    def sb = new StringBuilder()
    sb << '<entry xmlns="http://www.w3.org/2005/Atom" xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata" xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">'
    sb << '<content type="application/xml"><m:properties>'

    if (json instanceof Map) {
        json.each { k, v ->
            def key = k.toString()
            if (v == null) {
                sb << "<d:${key} m:null='true'/>"
            } else if (v instanceof Boolean) {
                sb << "<d:${key} m:type='Edm.Boolean'>${v}</d:${key}>"
            } else if (v instanceof Number) {
                sb << "<d:${key} m:type='Edm.Decimal'>${v}</d:${key}>"
            } else if (v instanceof String) {
                def esc = v.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;').replace("'",'&apos;')
                sb << "<d:${key}>${esc}</d:${key}>"
            } else {
                def text = v.toString()
                def esc = text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;').replace("'",'&apos;')
                sb << "<d:${key}>${esc}</d:${key}>"
            }
        }
    } else {
        def esc = body.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;').replace("'",'&apos;')
        sb << "<d:Value>${esc}</d:Value>"
    }

    sb << '</m:properties></content></entry>'
    message.setBody(sb.toString())
    return message
}
