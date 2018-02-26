# ### Audit and correct streetnames

# The first step is to look at the street names of the dataset, with the tag "addr:street". For this we take a list of the street types we expect to have for Perth:
# 
# expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
#             "Trail", "Highway", "Way", "Freeway", "Crossing", "Mall", "Loop", "Circle", "Crescent", "Gate", "Close",
#            "Mews", "Parade", "Terrace"]
# 
# And then we compare this list with all the street types that actually exist in our sample OSM for Perth. Each anomaly is included in the below dictionary with their occurence:

# In[9]:

# Create a regex for the street names as street_type_re 
# Create a default dictionary of standardized names
# Audit the file to find alternate names

OSMFILE = "perth_australia_sample.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
street_types = defaultdict(set)
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Highway", "Way", "Freeway", "Crossing", "Mall", "Loop", "Circle", "Crescent", "Gate", "Close",
           "Mews", "Parade", "Terrace"]

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile):
    osm_file = open(osmfile, "r")
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "way" or elem.tag == "node":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


# Run audit and print results
st_types = audit(OSMFILE)
pprint.pprint(dict(st_types))


# From the output of this audit, the data for Perth is actually pretty clean. There are only a couple of abbreviations to change (St and Ct) as well as a tag 'street' to change into 'Street' and 'Boulevarde' which is a typo for 'Boulevard'. I also find 'Subiaco' and 'Caversham', which are suburbs' names, 'Morrison' which is a Mall, 'Gelderland' which should be "Gelderland Entrance". Both Fairfield Garden and Connaught Garden should also have Garden spelled 'Gardens'.

# However, while the 50MB portion of the initial osm file looks pretty clean, we can have other cases of abbreviated names in the full dataset. Therefore, I'll use an extensive mapping of common abbreviations to update the names. The streetnames are returned below after correction, with the format "name => better_name"

# In[10]:

mapping = { "St": "Street",
            "St.": "Street",
            "ST": "Street",
            "st": "Street",
            "Rd.": "Road",
            "Rd": "Road",
            "RD": "Road",
            "Ave": "Avenue",
            "Ave.": "Avenue",
            "Blvd": "Boulevard",
            "BLVD": "Boulevard",
            "Cir": "Circle",
            "Ct": "Court",
            "Dr": "Drive",
            "Garden": "Gardens",
            "Trl": "Trail",
            "Ter": "Terrace",
            "Pl": "Place",
            "Pkwy": "Parkway",
            "Bnd": "Bend",
            "Mnr": "Manor",
            "Ln": "Lane",
            "street": "Street",
            "AVE": "Avenue",
            "Blvd.": "Boulevard",
            "Cirlce": "Circle",
            "DRIVE": "Drive",
            "Cv": "Cove",
            "Dr.": "Drive",
            "Druve": "Drive",
            "Holw": "Hollow",
            "Hwy": "Highway",
            "HWY": "Highway",
            "Pt": "Point",
            "Trce": "Trace",
            "ave": "Avenue",
            "Cres": "Crescent"
            }


# In[11]:

def update_name(name, mapping):
    """ Substitutes incorrect abbreviation with correct one. """
    m = street_type_re.search(name)
    if m:
        street_type = m.group()
        temp= 0
        try:
            temp = int(street_type)
        except:
            pass
        
        if street_type not in expected and temp == 0:
            try:
                name = re.sub(street_type_re, mapping[street_type], name)
            except:
                pass
    return name

for st_type, ways in st_types.iteritems():
    for name in ways:
        better_name = update_name(name, mapping)
        print name, "=>", better_name

