get_ipython().run_line_magic('run', 'Hooklog3.ipynb')

APIParDict = {
    # LIB
    "LoadLibrary": ["lpFileName", "Return"],
    
    # PROC
    "CreateProcess": ["lpApplicationName", "lpCommandLine", "lpCurrentDirectory", "Return"],
    "OpenProcess": ["dwDesiredAccess", "dwProcessId", "Return"],
    "ExitProcess": ["uExitCode", "Return"],
    "WinExec": ["lpCmdLine", "Return"],
    "CloseHandle": ["hObject", "Return"],
    "CreateRemoteThread": ["hProcess", "lpParameter", "dwCreationFlags", "lpThreadId", "Return"],
    "TerminateProcess": ["hProcess", "uExitCode", "Return"],
    "TerminateThread": ["hThread", "dwExitCode", "Return"],
    "CreateThread": ["lpStartAddress", "lpParameter", "dwCreationFlags", "lpThreadId", "Return"],
    "OpenThread": ["dwDesiredAccess", "dwThreadId", "Return"],
    
    # FILE
    "CopyFile": ["lpExistingFileName", "lpNewFileName", "Return"],
    "CreateFile": ["lpFileName", "dwDesiredAccess", "dwShareMode", "dwCreationDisposition", "Return"],
    "WriteFile": ["hFile", "Return"],
    "ReadFile": ["hFile", "Return"],
    "DeleteFile": ["lpFileName", "Return"],
    
    # REG
    "RegOpenKey": ["hKey", "lpSubKey", "phkresult", "Return"],
    "RegCloseKey": ["hKey", "Return"],
    "RegSetValue": ["hKey", "lpSubKey", "dwType", "lpData", "Return"],
    "RegCreateKey": ["hKey", "lpSubKey", "phkresult", "Return"],
    "RegDeleteKey": ["hKey", "lpSubKey", "Return"],
    "RegDeleteValue": ["hKey", "lpValueName", "Return"],
    "RegQueryValue": ["hKey", "lpValueName", "lpType", "lpData", "Return"],
    "RegEnumValue": ["hKey", "lpValueName", "lpType", "lpData", "Return"],
    
    # NET winhttp.dll
    "WinHttpConnect": ["pswzServerName", "nServerPort", "Return"],
    "WinHttpCreateUrl": ["pwszUrl", "Return"],
    "WinHttpOpen": ["pwszUserAgent", "Return"],
    "WinHttpOpenRequest": ["pwszObjectName", "Return"],
    "WinHttpReadData": ["lpBuffer", "Return"],
    "WinHttpSendRequest": ["pwszHeaders", "Return"],
    "WinHttpWriteData": ["lpBuffer", "Return"],
    "WinHttpGetProxyForUrl": ["lpcwszUrl", "Return"],
    
    # NET winnet32.dll
    "InternetOpen": ["lpszAgent", "Return"],
    "InternetConnect": ["lpszServerName", "Return"],
    "HttpSendRequest": ["lpszHeaders", "Return"],
    "GetUrlCacheEntryInfo": ["lpszUrlName", "Return"],
}

# -1:keep, 0:*, 1:1stLayer,10:negative, 20:postivite, 30:include
subPathDict = {
    "control panel": {'desktop': (-1,'desktop')},
    "system": {
        'setup': (-1,'sys_setup'),
        'controlset001':{'control':{'class': (0,'sys_ctlSet001_ctl_class\\*')}},
        'currentcontrolset':{
            'control':{
                'session manager': (0,'sys_curCtlSet_ctl_sessionManager\\*'),
                'mediaproperties': (0,'sys_curCtlSet_ctl_mediaProperties\\*'),
                'productoptions': (0,'sys_curCtlSet_ctl_productoptions\\*')},
            'services':{
                'netbt': (0,'sys_curCtlSet_svc_netbt\\*'),
                'tcpip': (10,'sys_curCtlSet_svc_tcpip', ['max','min','timeout']),
                'dnscache': (10,'sys_curCtlSet_svc_dnscache', ['max','min','timeout']),
                'ldap': (10,'sys_curCtlSet_svc_ldap', ['max','min','timeout']),
                'winsock': (10,'sys_curCtlSet_winsock', ['max','min','timeout']),
                'winsock2':{
                    'parameters':{
                        'protocol_catalog9': (0,'sys_curCtlSet_svc_winsock2_catalog9\\*'),
                        'namespace_catalog5': (0,'sys_curCtlSet_svc_winsock2_catalog5\\*'),
                        'ELSE' : (10,'sys_curCtlSet_svc_winsock2', ['max','min','timeout'])}, #
                    'ELSE' : (10,'sys_curCtlSet_svc_winsock2', ['max','min','timeout'])}, #
                'windows test 5.0': (1,'sys_curCtlSet_svc_winTest5.0')}},
        'ELSE':(-1,'sys')
    },
    'software':{
        'microsoft':{
            'commandprocessor': (0,'soft_ms_commandprocessor\\*'),
            'audiocompressionmanager': (0,'soft_ms_audiocompressionmanager\\*'),
            'multimedia':{
                'audio compression manager': (0,'soft_ms_mulitimedia_audiocompressionmanager\\*'),
                'ELSE' :(20,'soft_ms_multimedia',['audio'])},
            'ole': (0,'soft_ms_ole\\*'),
            'ctf': (0,'soft_ms_ctf\\*'),
            'com3': (0,'soft_ms_com3\\*'),
            'tracing': (1,'soft_ms_tracing'),
            'internet explorer':{'main':{'featurecontrol':(-1,'soft_ms_IE_featureCtl')}},
            'rpc': (10,'soft_ms_rpc', ['max','min','timeout']),
            'wbem':{
                'cimom': (20,'soft_ms_wbem_cimom', ['log']),
                'wmic': (0,'soft_ms_wbem_wmic\\*')},
            'windows nt':{'currentversion':{
                    'languagepack':(0,'soft_ms_winNT_languagepack\\*'),
                    'winlogon':(0,'soft_ms_winNT_winlogon\\*'),
                    'fontlink':(0,'soft_ms_winNT_fontlink\\*'),
                    'fontsubstitutes':(0,'soft_ms_winNT_fontsubstitutes\\*'),
                    'imm':(0,'soft_ms_winNT_imm\\*'),
                    'csdversion':(0,'soft_ms_winNT_csdversion\\*')}},
            'windows':{
                'currentversion':{
                    'thememanager':(0,'soft_ms_win_thememanager\\*'),
                    'telephony':(0,'soft_ms_win_telephony\\*'),
                    'profilelist':(0,'soft_ms_win_profilelist\\*'),
                    'programfilesdir':(0,'soft_ms_win_programfilesdir\\*'),
                    'commonfilesdir':(0,'soft_ms_win_commonfilesdir\\*'),
                    'explorer':{
                        'user shell folders': (0,'soft_ms_win_explorer_userShellFolder\\*'),
                        'shell folders': (0,'soft_ms_win_explorer_shellFolders\\*'),
                        'ELSE' : (-1,'soft_ms_win_explorer')},
                    'internet settings':{
                        'zones':(0,'soft_ms_win_internetSettings_zones\\*'),
                        'cache':{'paths':(0,'soft_ms_win_internetSettings_cache_paths\\*')},
                        'ELSE':(10,'soft_ms_win_internetSettings', ['max','min','timeout','length','bound','limit','timesecs','range'])},
                    'ELSE':(-1,'soft_ms_win_currentversion')
                }}
        }
    },
    'clsid':{'{REG}':{
            'inprocserver32':(0,'clsid_{REG}_inprocserver32\\*'),
            'appid':(0,'clsid_{REG}_appid\\*')
        }},
    'appid':{'{REG}':(0,'appid_{REG}\\*')},
    'interface':{'{REG}':{'proxystubclsid32':(0,'interface_{REG}_proxystubclsid32\\*')}},
    '{REG}':{
        'environment':(-1,'{REG}_envr'),
        'volatile environment':(-1,'{REG}_volatileEnvr')
    }
}

# -1:keep, 0:*, 1:1stLayer,10:negative, 20:postivite,
def subPathstrim(subpath): # -1, 0, 10, 20
    subFlag = False
    startIndex = 0
    startToken = ''
    returnVal = ''
    
    tokens = subpath.split('\\')
    for i, token in enumerate(tokens):
        if token in subPathDict:
            startIndex = i
            startToken = token
            break
        else:
            return ('\\').join(tokens)

    endIndex = startIndex
    endToken = startToken
    tmpDict = subPathDict
    endFlag = True if type(tmpDict[endToken])==tuple else False
    
    while( not endFlag ): # not to the end 
        tmpDict = tmpDict[endToken]
        endIndex += 1
        endToken = tokens[endIndex]
        if endToken in tmpDict:
            endFlag = True if type(tmpDict[endToken])==tuple else False
        else:
            endToken = 'ELSE'
            endIndex -= 1
            break
    
    #print subPath ####
    subType = tmpDict[endToken][0]
    subString = tmpDict[endToken][1]
    keepFrontString = ('\\').join(tokens[:startIndex])
    
    if subType == 0:
        returnVal = keepFrontString + subString
    elif subType == -1:
        keepEndtString = ('\\').join(tokens[endIndex+1:])
        returnVal = keepFrontString + subString + '\\' + keepEndtString
    elif subType == 1:
        keepEndtString = ('\\').join(tokens[endIndex+1:endIndex+2])
        returnVal = keepFrontString + subString + '\\' + keepEndtString
    elif subType == 10:
        ignoreli = tmpDict[endToken][2]
        keepEndtString = ('\\').join(tokens[endIndex+1:])
        keepFlag = True
        for k in ignoreli:
            if k in keepEndtString:
                returnVal = keepFrontString + subString + '\\*'
                keepFlag = False
                break
        if keepFlag:
            returnVal = keepFrontString + subString + '\\' + keepEndtString
    elif subType == 20:
        keepli = tmpDict[endToken][2]
        keepEndtString = ('\\').join(tokens[endIndex+1:])
        ignoreFlag = True
        for k in keepli:
            if k in keepEndtString:
                returnVal = keepFrontString + subString + '\\'+ k
                ignoreFlag = False
                break
        if ignoreFlag:
            returnVal = keepFrontString + subString + '\\*' 
    
    return returnVal

#MIKE: 20170713
def inKey(this_dict, myKey):
    return next((key for key in this_dict if key in myKey), False) # MIKE: key in mykey is correct!!


### MIKE: 20170714 new
def replace_strings(key):
    tokens = key.split('\\')

    rvalue = ""
    for token in tokens:
        
        # SID like token
        if token.count('-') > 3:
            rvalue += "\\{REG}"
        # file like token
        elif len(token.split('.')) >= 2 and token.split('.')[-1] in ["exe", "txt", "bat", "clb", "dll"] : #MIKE: 20170714, hack
            rvalue += "\\{FIL}." + token.split('.')[-1]
        # mist
        elif "mshist" in token:
            rvalue += "\\{MSHISTDATE}"
        else:
            rvalue += ('\\' + token)
            
    # MIKE: 20170714, there was a bug in old __remove_fileName(), it removes the first \\ accidentally
    # so I remove the first \\ here (so that subPathstrim() could work correctly).
    if rvalue.startswith('\\'): rvalue = rvalue[1:]
    if rvalue.startswith('\\'): rvalue = rvalue[1:]
            
    # sessionID
    if "software\\microsoft\\windows\\currentversion\\explorer\\sessioninfo" in rvalue:
        rvalue = rvalue[:rvalue.rindex('\\')] + "SESSIONID"

    return rvalue
###


def libTrans(value):
    global dir_dict
    if value == "": return "NON@NON@NON" # DIR@LIB@EXT
    
    DIR = LIB = EXT = "NON" # MIKE: 20170713, use capital
    
    lvalue = value.lower().replace('/', '\\')
    tokens = lvalue.split('\\')
    
    # DIR, MIKE: 20170714 change logic
    #if lvalue[1] != ':': DIR = "SYS"
    #elif lvalue[0] == '\\': DIR = "LOC" # MIKE: 20170713, really? != ??
    if len(tokens) == 1:
        DIR = "SYS"
    else:
        key = inKey(dir_dict, lvalue)
        DIR = dir_dict[key] if key else "ARB"
    
    # LIB
    LIB = tokens[-1].split('.')[0]
    
    # EXT
    t = tokens[-1].split('.')
    ext = t[-1]
    if ext == "" or len(t) == 1: # extension is . or no extension
        EXT = "DLL"
    else:
        EXT = ext.upper()
    
    return "@"+DIR+"@"+LIB+"@"+EXT


def execTrans(value):
    return fileTrans(value)

def fileTrans(value):
    global dir_dict
    if value == "": return "NON@NON"
    
    DIR = EXT = "NON" # MIKE: 20170713, use capital
    
    lvalue = value.lower().replace('/', '\\')
    tokens = lvalue.split('\\')
    
    # DIR, MIKE: 20170714 change logic
    if lvalue[:4] == ("\\\\.\\"):
        DIR = lvalue
    elif lvalue == 'conin$' or lvalue == 'conout$':
        DIR = 'CONSOLE'
    elif len(tokens) == 1:
        DIR = "LOC"
    else:
        key = inKey(dir_dict, lvalue)
        DIR = dir_dict[key] if key != False else "ARB"
    
    # EXT
    t = tokens[-1].split('.')
    ext = t[-1]
    if ext == "" or len(t) == 1: # extension is . or no extension
        EXT = "NON"
    else:
        EXT = ext.upper()

    return "@"+DIR+"@"+EXT

def shortenAccessModes(value):
    return "@"+value.replace(' ', ';')

def keySupPathTrans(value):
    keyBasic = keyTrans(value)
    HK = keyBasic[:keyBasic.index("@")]
    KEY = keyBasic[keyBasic.index("@")+1:]
    
    try:
        KEY = subPathstrim(KEY)
    except:
        KEY = KEY 
    return "@"+HK+"@"+KEY

# def keySupPathTransName(value):
    

def keyTrans(value):
    global hkey_dict
    if value == "": return "NON@NON"
    
    HK = KEY = "NON"
    
    lvalue = value.lower()
    tokens = lvalue.split('\\')
    
    # HK
    hkey = inKey(hkey_dict, tokens[0])
    HK = hkey_dict[hkey] if hkey else "SUBK"

    # KEY
    KEY = lvalue[lvalue.find('\\'):] if lvalue.find('\\') != -1 else lvalue
    KEY = replace_strings(KEY) # MIKE: 20170714, combine all rules

    return HK+"@"+KEY

def cmdTrans(value):
    return "@"+"CMD"

def handleTrans(value):
    return "@"+"HAND"

def pointerTrans(value):
    return "@"+"PTR"

def pidTrans(value):
    return "@"+"PID"

def tidTrans(value):
    return "@"+"TID"


dir_dict = {
    "\\windows\\system32\\": "SYS",
    "\\windows\\system\\": "SYS",
    "\\program files\\": "PRO",
    #"\\windows\\": "WIN",
    #"\\documents and settings\\all users\\": "USR",
    "\\documents and settings\\": "USR",
    "\\docume~1\\": "USR",
    "\\windows\\temp\\": "TMP",
    ":\\temp\\": "TMP"
}

hkey_dict = {
    "hkey_classes_root": "HKCR",
    "hkey_current_user": "HKCU",
    "hkey_local_machine": "HKLM",
    "hkey_users": "HKUS",
    "hkey_current_config": "HKCC"
}

funcDict = {
    # Path Trans
    "lpApplicationName": execTrans,
    "lpCommandLine": cmdTrans,
    "lpCmdLine": cmdTrans,
    "lpValueName": keySupPathTrans,
    "hKey": keySupPathTrans,
    "lpCurrentDirectory": keySupPathTrans,
    
    # File Trans
    "lpFileName": fileTrans,
    "lpExistingFileName": fileTrans,
    "lpNewFileName": fileTrans,
    
    # Handle or Addr value Trans
    # Assign constant value, because sometime these handle didn't traceback successfully
    "hObject": handleTrans, 
    "hFile": handleTrans,
    "hProcess": handleTrans,
    "hThread": handleTrans,
    "hProcess": handleTrans,
    "phkresult": pointerTrans,
    "lpStartAddress": pointerTrans,
    "lpParameter": pointerTrans,
    
    # ID Trans
    "dwProcessId": pidTrans,
    "lpThreadId": tidTrans,
    "dwThreadId": tidTrans,

    # string shorten
    "dwDesiredAccess": shortenAccessModes,
    "dwShareMode": shortenAccessModes
}

import os

class FeatureTrace(Hooklog3):
    
    def __init__(self, filePath):
        self.path = filePath
        self.par = True # overwrite it
        self.li = []#[] #list()
        self.length = 0
        self._parseDigitname()
        self._parseHooklog() # use Hooklog3's _parseHooklog
        
    def __str__(self):
        return "class FeatureTrace, %s, len = %d, digit name = %s" % (self.path, self.length, self.digitname)
    
    # use it if want to skip API
    def _skipAPI(self, api):
        return True if api not in APIParDict.keys() else False
    
    # read the parameter values in specific api
    def _getParValue(self, api, handle):
        parword = ""
        retword = ""
        
        while 1:
            pos = handle.tell() # record initial pos before going ahead
            
            line = handle.readline().decode('ISO 8859-1').strip() # read next line(usually params)
            if not line: # reach to the end of file
                break
            if line[0] == '#': # reach to next call
                handle.seek(pos) # back to initial position
                break
            delimiter = line.find('=') # MIKE: for '=' equal signal
            param = line[:delimiter].strip()
            value = line[delimiter+1:].strip()
            
            if self.isSelectedParameter(api, param): # check this parameter is selected or not
                if param == "Return":
                    retword += "#Ret#"
                    if value[0] == "S":
                        retword += "P" # positive
                    elif value[0] == "F":
                        retword += "N" # negative
                    elif value == "0":
                        retword += "0" # zero
                    else:
                        retVal = int(value, 16)
                        if retVal > 0:
                            retword += "P"
                        elif retVal < 0:
                            retword += "N"
                        else:
                            retword += "0"
                else:
                    value = self._parWinnowing(api, param, value)
                    parword += "#PR" + value
                    
        return api + parword + retword
                
    def isSelectedParameter(self, api, param):
        if param in APIParDict[api]:
            return True
        else:
            return False
        
    def _parWinnowing(self, api, param, value):
        if api == "LoadLibrary":
            return libTrans(value)
        else:
            return self._parTrans(param, value)
    
    def _parTrans(self, param, value):
        global funcDict
        return funcDict[param](value) if param in funcDict else "@"+value
    
    def getTrace_containTS(self): # return with Timestamp
        return self.li
    
    def getTrace_noContainTS(self): # return without Timestamp
        trace_noTS = [] #list()
        for (timestamp, api) in self.li:
            trace_noTS.append(api)
        
        return trace_noTS
    
    def getOriginalTrace_containTS(self):
        original_trace = [] #list()
        with open(self.path, 'rb') as handle:
            while 1:
                line = handle.readline().decode('ISO 8859-1').strip() # MIKE: 20170616, for python 3
                if not line: break # end of file
                if line[0] == '#': # start a new call
                    tick = line[1:].strip()
                    api = handle.readline().decode('ISO 8859-1').strip() # MIKE: 20170616, for python 3
                    
                    if self._skipAPI(api): continue
                    
                    api = self._getOriginalParValue(api, handle)
                    original_trace.append((tick, api))
            
            handle.close()
            
        original_trace.sort(key = lambda tup: tup[0]) # sort by tick
        
        return original_trace
    
    def getOriginalTrace_withoutTS(self):
        trace_withoutTS = [] #list()
        for timestamp, api in self.getOriginalTrace_containTS():
            trace_withoutTS.append(api)
        
        return trace_withoutTS
    
    def _getOriginalParValue(self, api, handle):
        result = ""
        
        while 1:
            pos = handle.tell() # record initial pos before going ahead
            
            line = handle.readline().decode('ISO 8859-1').strip() # read next line(usually params)
            if not line: # reach to the end of file
                break
            if line[0] == '#': # reach to next call
                handle.seek(pos) # back to initial position
                break
            delimiter = line.find('=') # MIKE: for '=' equal signal
            param = line[:delimiter].strip()
            value = line[delimiter+1:].strip()
            
            if self.isSelectedParameter(api, param): # check this parameter is selected or not
                if param == 'lpData' and result[-12:-2]=="REG_BINARY":
                    line = "lpData=BinaryString"
                result += line.strip() + "; "
                    
        return api + ': ' + result

