<!-- transform.xsl -->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
    <xsl:output method="html" indent="yes"/>
    
    <!-- Define the word to highlight as a parameter -->
    <xsl:param name="highlight-word-1" select="'AlignMe'"/>  <!-- true match 1 -->
	<xsl:param name="highlight-word-2" select="'HMMER'"/>  <!-- true match 2 -->
	<xsl:param name="highlight-word-3" select="'alignme'"/>  <!-- false match -->


    <xsl:template match="/">
        <html>
            <head>
                <title><xsl:value-of select="article/front/article-meta/title-group/article-title"/></title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; }
                    .highlight1 { background-color: rgb(127, 255, 127); font-weight: normal; }
					.highlight2 { background-color: rgb(147, 147, 255); font-weight: normal; }
					.highlight3 { background-color: rgb(255, 127, 127); font-weight: normal; }
                </style>
            </head>
            <body>
                <h1><xsl:value-of select="article/front/article-meta/title-group/article-title"/></h1>
                <xsl:apply-templates select="article/body/sec"/>
            </body>
        </html>
    </xsl:template>
    
    <!-- Process section elements -->
    <xsl:template match="sec">
        <section>
            <h2><xsl:value-of select="title"/></h2>
            <xsl:apply-templates select="p"/>
        </section>
    </xsl:template>
    
    <!-- Process paragraph elements -->
    <xsl:template match="p">
        <p>
            <xsl:apply-templates/>
        </p>
    </xsl:template>

    <!-- Template to highlight the specified word in text nodes -->
    <xsl:template match="text()">
        <xsl:variable name="text" select="."/>
        
        <!-- Loop to find and highlight each occurrence of the word -->
        <xsl:choose>
            <xsl:when test="contains($text, $highlight-word-1)">
                <!-- Part before the word -->
                <xsl:value-of select="substring-before($text, $highlight-word-1)"/>
                
                <!-- The word itself, wrapped in a span for highlighting -->
                <span class="highlight1">
                    <xsl:value-of select="$highlight-word-1"/>
                </span>
                
                <!-- Process the remaining text after the word recursively -->
                <xsl:variable name="remaining-text" select="substring-after($text, $highlight-word-1)"/>
                <xsl:call-template name="highlight-text">
                    <xsl:with-param name="text" select="$remaining-text"/>
                </xsl:call-template>
            </xsl:when>
			<xsl:when test="contains($text, $highlight-word-2)">
                <!-- Part before the word -->
                <xsl:value-of select="substring-before($text, $highlight-word-2)"/>
                
                <!-- The word itself, wrapped in a span for highlighting -->
                <span class="highlight2">
                    <xsl:value-of select="$highlight-word-2"/>
                </span>
                
                <!-- Process the remaining text after the word recursively -->
                <xsl:variable name="remaining-text" select="substring-after($text, $highlight-word-2)"/>
                <xsl:call-template name="highlight-text">
                    <xsl:with-param name="text" select="$remaining-text"/>
                </xsl:call-template>
            </xsl:when>
			<xsl:when test="contains($text, $highlight-word-3)">
                <!-- Part before the word -->
                <xsl:value-of select="substring-before($text, $highlight-word-3)"/>
                
                <!-- The word itself, wrapped in a span for highlighting -->
                <span class="highlight3">
                    <xsl:value-of select="$highlight-word-3"/>
                </span>
                
                <!-- Process the remaining text after the word recursively -->
                <xsl:variable name="remaining-text" select="substring-after($text, $highlight-word-3)"/>
                <xsl:call-template name="highlight-text">
                    <xsl:with-param name="text" select="$remaining-text"/>
                </xsl:call-template>
            </xsl:when>
            
            <!-- If the text doesn’t contain the word, output as-is -->
            <xsl:otherwise>
                <xsl:value-of select="$text"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

    <!-- Recursive template to continue highlighting occurrences in the remaining text -->
    <xsl:template name="highlight-text">
        <xsl:param name="text"/>
        <xsl:choose>   
			<xsl:when test="contains($text, $highlight-word-1)">
                <!-- Part before the word -->
                <xsl:value-of select="substring-before($text, $highlight-word-1)"/>
                
                <!-- The word itself, wrapped in a span for highlighting -->
                <span class="highlight">
                    <xsl:value-of select="$highlight-word-1"/>
                </span>
                
                <!-- Process the next part recursively -->
                <xsl:variable name="remaining-text" select="substring-after($text, $highlight-word-1)"/>
                <xsl:call-template name="highlight-text">
                    <xsl:with-param name="text" select="$remaining-text"/>
                </xsl:call-template>
            </xsl:when>
            
            <!-- Output the remaining text if there are no more occurrences -->
            <xsl:otherwise>
                <xsl:value-of select="$text"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>
</xsl:stylesheet>
