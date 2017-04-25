
# coding: utf-8

# In[1]:




# In[1]:




# In[42]:

"""
Your task in this exercise has two steps:

- audit the OSMFILE and change the variable 'mapping' to reflect the changes needed to fix 
    the unexpected street types to the appropriate ones in the expected list.
    You have to add mappings only for the actual problems you find in this OSMFILE,
    not a generalized solution, since that may and will depend on the particular area you are auditing.
- write the update_name function, to actually fix the street name.
    The function takes a string with street name as an argument and should return the fixed name
    We have provided a simple test so that you see what exactly is expected
"""
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "C:\Users\Fareeda\P3\melbourne_australia.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
street_dirc_re = re.compile(r'(\b\S+) (\b\S+) (\b\S+)', re.IGNORECASE)
stat_type_re = re.compile(r'[a-z]', re.IGNORECASE)

#The Expected street type
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road","Parade", 
            "Trail", "Parkway", "Commons","Circle","Circuit","Highway","Grove","Close","Terrace","Crescent","Way","Gardens"]

dir_expected = ["North","West","East","South"]
#state expected[]
# UPDATE THIS VARIABLE
mapping = { "St": "Street",
            "St.": "Street",
             "Stree":"Street",
             "Ave": "Avenue",
            "Rd.": "Road"
            }
state_exp=["Victoria"]
stat_mapping={
     "VIC": "Victoria",
     "VICTORIA": "Victoria",
     "Melbourne":  "Victoria"

}

## audit the street nam:
def audit_street_type(street_types, street_name):
        m = street_type_re.search(street_name)
        if m:
            street_type = m.group()
            if street_type not in expected:
                if street_type in dir_expected:
                    update_name(street_name)
                else:
                    street_types[street_type].add(street_name)



def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = "C:\Users\Fareeda\P3\melbourne_australia.osm"
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    return street_types


def update_name(name):
    street = street_dirc_re.search(name)
    name = street_dirc_re.sub(r'\3 \1 \2', name)
    return name

#Audit the state name:
def audit_state(osmfile):
       for event, element in ET.iterparse(osmfile, events=("start","end")):
        if element.tag == "node" or element.tag == "way":
            for tag in element.iter("tag"):
                if is_state(tag):
                    if state_exp[0].lower() != tag.attrib['v'].lower():
                        if tag.attrib['v'].strip() in stat_mapping:
                            tag.attrib['v'] = updat_state_name(tag.attrib['v'].strip(), stat_mapping)
    
def is_state(elem):
    
    return (elem.attrib['k'] == "addr:state")
    

def updat_state_name(name,stat_mapping):
    if name in stat_mapping.keys(): 
        print name
        name = stat_mapping[name]
        return name

audit_state("C:\Users\Fareeda\P3\melbourne_australia.osm")

#def test():
#    st_types = audit("C:\Users\Fareeda\sample.osm")
#    pprint.pprint(dict(st_types))

#if __name__ == '__main__':
#    test()


# In[ ]:




# In[ ]:





# In[ ]:




# In[ ]:




# In[ ]:



