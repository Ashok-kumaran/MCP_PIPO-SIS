<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="xml" indent="yes"/>
  <xsl:template match="/">
    <A_SalesOrder>
      <xsl:apply-templates select="//*"/>
    </A_SalesOrder>
  </xsl:template>
  <!-- Map common fields if present in JSON-XML -->
  <xsl:template match="*">
    <xsl:choose>
      <xsl:when test="name()='SalesOrderType'">
        <SalesOrderType><xsl:value-of select="text()"/></SalesOrderType>
      </xsl:when>
      <xsl:when test="name()='SalesOrganization'">
        <SalesOrganization><xsl:value-of select="text()"/></SalesOrganization>
      </xsl:when>
      <xsl:when test="name()='DistributionChannel'">
        <DistributionChannel><xsl:value-of select="text()"/></DistributionChannel>
      </xsl:when>
      <xsl:when test="name()='OrganizationDivision'">
        <OrganizationDivision><xsl:value-of select="text()"/></OrganizationDivision>
      </xsl:when>
      <xsl:when test="name()='SoldToParty'">
        <SoldToParty><xsl:value-of select="text()"/></SoldToParty>
      </xsl:when>
      <xsl:when test="name()='PurchaseOrderByCustomer'">
        <PurchaseOrderByCustomer><xsl:value-of select="text()"/></PurchaseOrderByCustomer>
      </xsl:when>
      <xsl:when test="name()='RequestedDeliveryDate'">
        <RequestedDeliveryDate><xsl:value-of select="text()"/></RequestedDeliveryDate>
      </xsl:when>
      <!-- Ignore keys and wrapper nodes produced by JSON-XML converter -->
      <xsl:otherwise/>
    </xsl:choose>
  </xsl:template>
</xsl:stylesheet>
