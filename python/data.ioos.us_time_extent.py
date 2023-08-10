import requests, json

headers = {'Content-Type': 'application/xml'}

input='''
<csw:GetRecords xmlns:csw="http://www.opengis.net/cat/csw/2.0.2"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:ogc="http://www.opengis.net/ogc"
    xmlns:gml="http://www.opengis.net/gml" outputSchema="http://www.opengis.net/cat/csw/2.0.2"
    outputFormat="application/xml" version="2.0.2" service="CSW" resultType="results"
    maxRecords="1000"
    xsi:schemaLocation="http://www.opengis.net/cat/csw/2.0.2 http://schemas.opengis.net/csw/2.0.2/CSW-discovery.xsd">
    <csw:Query typeNames="csw:Record">
        <csw:ElementSetName>full</csw:ElementSetName>
        <csw:Constraint version="1.1.0">
            <ogc:Filter>
                <ogc:And>
                    <ogc:BBOX>
                        <ogc:PropertyName>ows:BoundingBox</ogc:PropertyName>
                        <gml:Envelope srsName="urn:ogc:def:crs:OGC:1.3:CRS84">
                            <gml:lowerCorner> -158.4 20.7</gml:lowerCorner>
                            <gml:upperCorner> -157.2 21.6</gml:upperCorner>
                        </gml:Envelope>
                    </ogc:BBOX>
                    <ogc:PropertyIsLessThanOrEqualTo>
                        <ogc:PropertyName>apiso:TempExtent_begin</ogc:PropertyName>
                        <ogc:Literal>2016-12-01T16:43:00Z</ogc:Literal>
                    </ogc:PropertyIsLessThanOrEqualTo>
                    <ogc:PropertyIsGreaterThanOrEqualTo>
                        <ogc:PropertyName>apiso:TempExtent_end</ogc:PropertyName>
                        <ogc:Literal>2014-12-01T16:43:00Z</ogc:Literal>
                    </ogc:PropertyIsGreaterThanOrEqualTo>
                    <ogc:PropertyIsLike wildCard="*" singleChar="?" escapeChar="\\">
                        <ogc:PropertyName>apiso:AnyText</ogc:PropertyName>
                        <ogc:Literal>*G1SST*</ogc:Literal>
                    </ogc:PropertyIsLike>
                </ogc:And>
            </ogc:Filter>
        </csw:Constraint>
    </csw:Query>
</csw:GetRecords>
''';

endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw'
xml_string=requests.post(endpoint, data=input, headers=headers).text
print(xml_string[:2000])

endpoint = 'https://dev-catalog.ioos.us/csw'
xml_string=requests.post(endpoint, data=input, headers=headers).text
print(xml_string[:2000])

input='''
<csw:GetRecords xmlns:csw="http://www.opengis.net/cat/csw/2.0.2"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:ogc="http://www.opengis.net/ogc"
    xmlns:gml="http://www.opengis.net/gml" outputSchema="http://www.opengis.net/cat/csw/2.0.2"
    outputFormat="application/xml" version="2.0.2" service="CSW" resultType="results"
    maxRecords="1000"
    xsi:schemaLocation="http://www.opengis.net/cat/csw/2.0.2 http://schemas.opengis.net/csw/2.0.2/CSW-discovery.xsd">
    <csw:Query typeNames="csw:Record">
        <csw:ElementSetName>full</csw:ElementSetName>
        <csw:Constraint version="1.1.0">
            <ogc:Filter>
                <ogc:And>
                    <ogc:BBOX>
                        <ogc:PropertyName>ows:BoundingBox</ogc:PropertyName>
                        <gml:Envelope srsName="urn:ogc:def:crs:OGC:1.3:CRS84">
                            <gml:lowerCorner> -158.4 20.7</gml:lowerCorner>
                            <gml:upperCorner> -157.2 21.6</gml:upperCorner>
                        </gml:Envelope>
                    </ogc:BBOX>
                    <ogc:PropertyIsLike wildCard="*" singleChar="?" escapeChar="\\">
                        <ogc:PropertyName>apiso:AnyText</ogc:PropertyName>
                        <ogc:Literal>*G1SST*</ogc:Literal>
                    </ogc:PropertyIsLike>
                </ogc:And>
            </ogc:Filter>
        </csw:Constraint>
    </csw:Query>
</csw:GetRecords>
''';

endpoint = 'https://dev-catalog.ioos.us/csw'
xml_string=requests.post(endpoint, data=input, headers=headers).text
print(xml_string[:2000])



