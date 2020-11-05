##import dnspython as dns
##import dns.resolver
import socket
##import whois
import os
import requests

def get_whois(url):
    command = "whois " + url
    process = os.popen(command)
    results = str(process.read())
    return results

exampleDomain = 'w-hs.de'

addr1 = socket.gethostbyname(exampleDomain)
print("------------------------------------------")
print("IP: Addresse von " + exampleDomain)
print(addr1)


##result = dns.resolver.query('www.w-hs.de', 'A')
for ipval in result:
    print('IP', ipval.to_text())
print("------------------------------------------")


##result = dns.resolver.query('mail.google.com', 'CNAME')
for cnameval in result:
    print('cname target address:', cnameval.target)
print("------------------------------------------")
#result = dns.resolver.query('mail.google.com', 'MX')
#for exdata in result:
#    print ('MX Record:', exdata.exchange.text())


print(get_whois('w-hs.de'))
print("------------------------------------------")
print(get_whois('134.119.224.183'))
print("------------------------------------------")

def ScanDomain():
    domain = "google.com"
    # read all subdomains
    file = open("subdomains-100.txt")
    # read all content
    content = file.read()
    # split by new lines
    subdomains = content.splitlines()


    for subdomain in subdomains:
        # construct the url
        url = f"http://{subdomain}.{domain}"
        try:
            # if this raises an ERROR, that means the subdomain does not exist
            requests.get(url)
        except requests.ConnectionError:
            # if the subdomain does not exist, just pass, print nothing
            pass
        else:
            print("[+] Discovered subdomain:", url)

ScanDomain()