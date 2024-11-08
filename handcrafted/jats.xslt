<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
    <xsl:output method="html" indent="yes"/>

    <xsl:template match="/">
        <html>
            <head>
                <title>
                    <xsl:value-of select="//article-title"/>
                </title>
            </head>
            <body>
                <h1>
                    <xsl:value-of select="//article-title"/>
                </h1>
                <div>
                    <xsl:apply-templates select="//front"/>
                </div>
                <div>
                    <xsl:apply-templates select="//body"/>
                </div>
                <div>
                    <xsl:apply-templates select="//back"/>
                </div>
            </body>
        </html>
    </xsl:template>

    <xsl:template match="front">
        <xsl:for-each select="node()[not(self::article-meta)]">
            <p>
                <xsl:call-template name="concat-text-nodes">
                    <xsl:with-param name="nodes" select=".//text()" />
                </xsl:call-template>
            </p>
        </xsl:for-each>
        <xsl:apply-templates select="article-meta"/>
    </xsl:template>

    <xsl:template match="article-meta">
        <xsl:for-each select="node()">
            <p>
                <xsl:call-template name="concat-text-nodes">
                    <xsl:with-param name="nodes" select=".//text()" />
                </xsl:call-template>
            </p>
        </xsl:for-each>
    </xsl:template>

    <xsl:template match="body | sec">
        <h2>
            <xsl:value-of select="title"/>
        </h2>
        <xsl:for-each select="node()[not(self::title or self::sec)]">
            <p>
                <xsl:call-template name="concat-text-nodes">
                    <xsl:with-param name="nodes" select=".//text()" />
                </xsl:call-template>
            </p>
        </xsl:for-each>
        <xsl:apply-templates select="sec"/>
    </xsl:template>

    <xsl:template match="back">
        <h2>
            <xsl:value-of select="title"/>
        </h2>
        <xsl:for-each select="node()[not(self::title or self::sec or self::ref-list)]">
            <p>
                <xsl:call-template name="concat-text-nodes">
                    <xsl:with-param name="nodes" select=".//text()" />
                </xsl:call-template>
            </p>
        </xsl:for-each>
        <xsl:apply-templates select="sec"/>
        <xsl:apply-templates select="ref-list"/>
    </xsl:template>

    <xsl:template match="ref-list">
        <h2>
            <xsl:value-of select="title"/>
        </h2>
        <xsl:for-each select="node()[not(self::title or self::ref)]">
            <p>
                <xsl:call-template name="concat-text-nodes">
                    <xsl:with-param name="nodes" select=".//text()" />
                </xsl:call-template>
            </p>
        </xsl:for-each>
        <ul>
            <xsl:for-each select="ref">
                <li>
                    <xsl:call-template name="concat-text-nodes">
                        <xsl:with-param name="nodes" select=".//text()" />
                    </xsl:call-template>
                </li>
            </xsl:for-each>
        </ul>
    </xsl:template>

    <!-- Recursive template to handle text nodes with spaces (ChatGPT) -->
    <xsl:template name="concat-text-nodes">
        <xsl:param name="nodes" />
        <xsl:param name="index" select="1" />

        <xsl:choose>
            <xsl:when test="$index &lt;= count($nodes)">
                <xsl:value-of select="$nodes[$index]" />
                <xsl:if test="$index &lt; count($nodes)">
                    <xsl:text> </xsl:text>
                </xsl:if>
                <xsl:call-template name="concat-text-nodes">
                    <xsl:with-param name="nodes" select="$nodes" />
                    <xsl:with-param name="index" select="$index + 1" />
                </xsl:call-template>
            </xsl:when>
        </xsl:choose>
    </xsl:template>
</xsl:stylesheet>
