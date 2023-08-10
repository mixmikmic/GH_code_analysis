class View:
    def __init__(self, Id, Name):
        self.id = Id
        self.name = Name
        self.children = []

    def getID(self):
        return self.id

    def getName(self):
        return self.name

    def addChild(self, child):
        self.children.append(child)

    def removeChild(self, child):
        self.children.remove(child)

    def printDetails(self):
        print("%s - %s " % (self.id, self.name))
        for child in self.children:
            print("\t")
            child.printDetails()

class Category(View):
    def __init__(self, Id, Name, Desc):
        View.__init__(self, Id, Name)
        self.description = Desc

    def getDesc():
        return self.description

class MetaAttackPattern(Category):
    def __init__(self, Id, Name, Desc):
        Category.__init__(self, Id, Name, Desc)
        self.fields = Fields()

class StandardAttackPattern(MetaAttackPattern):
    def __init__(self, Id, Name, Desc):
        MetaAttackPattern.__init__(self, Id, Name, Desc)

class DetailedAttackPattern(StandardAttackPattern):
    def __init__(self, Id, Name, Desc):
        StandardAttackPattern.__init__(self, Id, Name, Desc)

class Fields():
    def __init__(self):
        self.attack_prerequisites = []
        self.typical_severity = ""
        self.typical_likelihood_of_exploit = ""
        self.methods_of_attack = []
        self.example_instance = ""
        self.attacker_skill_or_knowledge_required = []
        self.resources_required = ""
        self.probing_techniques = []
        self.indicators_warnings_of_attack = []
        self.obfuscation_techniques = []
        self.solutions_or_mitigations = []
        self.attack_motivation_consequences = []
        self.injection_vector = ""
        self.payload = ""
        self.activation_zone = ""
        self.payload_activation_impact = ""
        self.related_weaknesses = []
        self.related_attack_patterns = []
        self.related_vulnerabilities = []
        self.relevant_security_requirements = []
        self.related_security_principles = []
        self.related_guidelines = []
        self.purposes = []
        self.cia_impact = {"Confidentiality_Impact" : None, "Integrity_Impact" : None, "Availability_Impact" : None}

    def addAttackPrerequisite(self, value):
        self.attack_prerequisites.append(value)

    def getAttackPrerequisite(self):
        return self.attack_prerequisite

    def setTypicalSeverity(self, value):
        self.typical_severity = value

    def getTypicalSeverity(self):
        return self.typical_severity

    def setTypicalLikelihoodOfExploit(self, value):
        self.typical_likelihood_of_exploit = value

    def getTypicalLikelihookOfExploit(self):
        return self.typical_likelihood_of_exploit

    def addMethodOfAttack(self, value):
        self.methods_of_attack.append(value)

    def getMethodOfAttack(self):
        return self.method_of_attack

    def setExampleInstance(self, value):
        self.example_instance = value

    def getExampleInstance(self):
        return self.example_instance

    def addAttackerSkillsOrKnowledgeRequired(self, value1, value2):
        temp_dict = dict()
        temp_dict["Skill_or_Knowledge_Level"] = value1
        temp_dict["Skill_or_Knowledge_Type"] = value2
        self.attacker_skill_or_knowledge_required.append(temp_dict)

    def getAttackerSkillOrKnowledgeRequired(self):
        return self.attacker_skill_or_knowledge_required

    def setResourcesRequired(self, value):
        self.resources_required = value

    def getResourcesRequired(self):
        return self.resources_required

    def addProbingTechnique(self, value):
        self.probing_techniques.append(value)

    def getProbingTechniques(self):
        return self.probing_techniquesd

    def addIndicatorsWarningsOfAttack(self, value):
        self.indicators_warnings_of_attack.append(value)

    def getIndicatorsWarningsOfAttack(self):
        return self.indicators_warnings_of_attack

    def addObfuscationTechnique(self, value):
        self.obfuscation_techniques.append(value)

    def getObfuscationTechniques(self):
        return self.obfuscation_techniques

    def addSolutionsOrMitigations(self, value):
        self.solutions_or_mitigations.append(value)

    def getSolutionsOrMitigations(self):
        return self.solutions_or_mitigations

    def addAttackMotivationConsequences(self, value1, value2, value3):
        temp_dict = dict()
        temp_dict["Consequence_Scope"] = value1
        temp_dict["Consequence_Technical_Impact"] = value2
        temp_dict["Consequence_Note"] = value3
        self.attack_motivation_consequences.append(temp_dict)

    def getAttackMotivationConsequences(self):
        return self.attack_motivation_consequences

    def setInjectionVector(self, value):
        self.injection_vector = value

    def getInjectionVector(self):
        return self.injection_vector

    def setPayload(self, value):
        self.payload = value

    def getPayload(self):
        return self.payload

    def setActivationZone(self, value):
        self.activation_zone = value

    def getActivationZone(self):
        return self.activation_zone

    def setPayloadActivationImpact(self, value):
        self.payload_activation_impact = value

    def getPayloadActivationImpact(self):
        return self.payload_activation_impact

    def addRelatedVulnerability(self, value1, value2):
        temp_dict = dict()
        temp_dict["Vulnerability_ID"] = value1
        temp_dict["Vulnerability_Description"] = value2
        self.related_weaknesses.append(temp_dict)

    def getRelatedVulnerabilities(self):
        return self.related_vulnerabilities

    def addRelatedWeaknesses(self, value1, value2):
        temp_dict = dict()
        temp_dict["CWE_ID"] = value1
        temp_dict["Weakness_Relationship_Type"] = value2
        self.related_weaknesses.append(temp_dict)

    def getRelatedWeaknesses(self):
        return self.related_weaknesses

    def addRelatedAttackPattern(self, value1, value2, value3, value4):
        temp_dict = dict()
        temp_dict["Relationship_View_ID"] = value1
        temp_dict["Relationship_Target_Form"] = value2
        temp_dict["Relationship_Nature"] = value3
        temp_dict["Relationship_Target_ID"] = value4
        self.related_attack_patterns.append(temp_dict)

    def getRelatedAttackPatterns(self):
        return self.related_attack_patterns

    def addRelatedSecurityPrinciple(self, value):
        self.related_security_principles.append(value)

    def getRelatedSecurityPrinciples(self):
        return self.related_security_principles

    def addRelevantSecurityRequirement(self, value):
        self.relevant_security_requirements.append(value)

    def getRelevantSecurityRequirements(self):
        return self.relevant_security_requirements

    def addRelatedGuideline(self, value):
        self.related_guidelines.append(value)

    def getRelatedGuidelines(self):
        return self.related_guidelines

    def addPurpose(self, value):
        self.purposes.append(value)

    def getPurposes(self):
        return self.purposes

    def setCIAImpact(self, value1, value2, value3):
        self.cia_impact["Confidentiality_Impact"] = value1
        self.cia_impact["Integrity_Impact"] = value2
        self.cia_impact["Availability_Impact"] = value3

    def getCIAImpact(self):
        return self.cia_impact

