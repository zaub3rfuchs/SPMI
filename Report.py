import os
import threading
import time


##################################################################################################################
###                                               CLASSES                                                      ### 
##################################################################################################################

class Report():
    def __init__(self, ttl):
        self.ttl = ttl

class MXRecord(Report):
    def __init__(self, ttl, hostname):
        self.ttl = ttl
        self.hostname = hostname

class ARecord(Report):
    def __init__(self, ttl, ipV4, whoIsQuery):
        self.ttl = ttl
        self.ipV4 = ipV4
        self.whoIsQuery = whoIsQuery

class AAAARecord(Report):
    def __init__(self, ttl, ipV6, whoIsQuery):
        self.ttl = ttl
        self.ipV6 = ipV6
        self.whoIsQuery = whoIsQuery

class NSRecord(Report):
    def __init__(self, ttl, nsDomain):
        self.ttl = ttl
        self.nsDomain = nsDomain

class TXTRecord(Report):
    def __init__(self, ttl, txt):
        self.ttl = ttl
        self.txt = txt


##################################################################################################################
###                                               DEFINITIONS                                                  ### 
##################################################################################################################

recordDict = {}

def digRecord(url, type):
    ## hier könnte man eventuel noch mit splitlines arbeiten um Dig any zu benutzen
    command = "dig +nocmd +noall +answer " + url + " " + type ##| cut -f3"
    process = os.popen(command)
    results = str(process.read())
    return results

def createRecord(type, record):

    lines = record.splitlines()
    list = []

    if type == "MX":
        for line in lines:
            ttl = line.split()[1]
            hostname = line.split()[5]
            list.append(MXRecord(ttl, hostname))
    elif type == "NS":
        for line in lines:
            ttl = line.split()[1]
            nsDomain = line.split()[4]
            list.append(NSRecord(ttl, nsDomain))
    elif type == "A":
        for line in lines:
            ttl = line.split()[1]
            ipV4 = line.split()[4]
            list.append(ARecord(ttl, ipV4, "platzhalter"))
    elif type == "AAAA":
        for line in lines:
            ttl = line.split()[1]
            ipV6 = line.split()[4]
            list.append(AAAARecord(ttl, ipV6, "platzhalter"))
    elif type == "TXT": ## manche txt records haben leerzeichen damit funktioniert der split befehl nicht mehr
        for line in lines:
            ttl = line.split()[1]
            txt = line.split()[4].replace('"', '')
            list.append(TXTRecord(ttl, txt))
    
    recordDict[" " + type] = list
    return recordDict[" " + type]
    
def work(url, type):
    record = digRecord(url, type)
    return createRecord(type, record)


##################################################################################################################
###                                               SERIAL                                                       ### 
##################################################################################################################

startSerial = time.perf_counter()

recordTypesSerial = ["MX", "A", "AAAA", "NS", "TXT"]
urlSerial = "w-hs.de"

for recordType in recordTypesSerial:
    work(urlSerial, recordType)
finishSerial = time.perf_counter()

print(f'Finishes in {round(finishSerial-startSerial, 2)} second(s) [Serial]')


##################################################################################################################
###                                               THREADS                                                      ### 
##################################################################################################################

start = time.perf_counter()

threads = []
recordTypes = ["MX", "A", "AAAA", "NS", "TXT"]
url = "w-hs.de"

for recordType in recordTypes:
    t = threading.Thread(target=work, args=[url, recordType])
    t.start()
    threads.append(t)

for thread in threads:
    thread.join()

finish = time.perf_counter()
print(f'Finishes in {round(finish-start, 2)} second(s) [Multithreaded]')


##################################################################################################################
###                                               TESTING                                                      ### 
##################################################################################################################

print("------------------------------------------------------------------------------------------------------------")
print(digRecord("w-hs.de", "any"))