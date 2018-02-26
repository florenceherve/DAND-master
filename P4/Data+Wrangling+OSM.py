
# coding: utf-8

# # Data Wrangling - Perth, Australia

# The dataset was downloaded from Mapzen: https://mapzen.com/data/metro-extracts/metro/perth_australia/
# 
# This is a metro extract of Perth, WA, Australia. Size of 254MB with the remaining sample file to be ~50MB.
# 
# I've chosen Perth as I have been living there briefly when I was a student, and have some fond memories of it.

# ## Resources

# - code from the course Case Sudy
# - blog posts from discussions.udacity.com
# - for the conversion from XLM to CSV and CSV to SQL I used the following GitHub as a reference: https://gist.github.com/swwelch/f1144229848b407e0a5d13fcb7fbbd6f

# ## Getting started

# In[1]:

# Hide code cells

from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# In[2]:

# Set up libraries
import xml.etree.cElementTree as ET
import pprint
import re
from collections import defaultdict
import csv
import codecs

import cerberus
import schema

import os
from hurry.filesize import size


# In[3]:

# %load 'schema.py'


# Once downloaded and unzipped, the OSM file for Perth, Australia has a size 254 MB.
# The requirement for this project is a size of 50 MB. Hence I am using the code provided in the Project Overview to create a sample file, by iterating over the lines by k buckets.

# In[ ]:

#!/usr/bin/env python

from pprint import pprint
import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow

OSM_FILE = "perth_australia.osm"
SAMPLE_FILE = "perth_australia_sample.osm"

k = 5

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag

    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()
        
        
with open(SAMPLE_FILE, 'wb') as output:
    output.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write(b'<osm>\n  ')

    # Write every kth top level element
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8'))

    output.write(b'</osm>')


# The resulting sample file has a size of 51.5 MB.

# In[5]:

# opening file in filename
filename = open("perth_australia_sample.osm", "r")


# ## Data Audit

# ### Tag types

# Count of each of the tags in the OSM data: 

# In[6]:

# Iterative parsing from the problem set in the course
"""
Your task is to use the iterative parsing to process the map file and find out not only what tags are there, but also how many, 
to get the feeling on how much of which data you can expect to have in the map.
Fill out the count_tags function. It should return a dictionary with the tag name as the key and number of times this tag can be
encountered in the map as value.
"""
def count_tags(samplefile):
    tags = {}
    for event, element in ET.iterparse(samplefile):
        if element.tag not in tags.keys():
            tags[element.tag] = 1
        else:
            tags[element.tag] += 1
    return tags

count_tags(filename)


# We would like to change the data model and expand the "addr:street" type of keys to a dictionary like this:
# {"address": {"street": "Some value"}}
# So, we have to see if we have such tags, and if we have any tags with problematic characters.
# 
# Below we have a count of each of four tag categories in a dictionary:
#   "lower", for tags that contain only lowercase letters and are valid,
#   "lower_colon", for otherwise valid tags with a colon in their names,
#   "problemchars", for tags with problematic characters, and
#   "other", for other tags that do not fall into the other three categories.

# In[7]:

# Tag types from the problem set in the course
"""
Your task is to explore the data a bit more. Before you process the data and add it into your database, you should check the
"k" value for each "<tag>" and see if there are any potential problems.

We have provided you with 3 regular expressions to check for certain patterns in the tags. As we saw in the quiz earlier, 
we would like to change the data model and expand the "addr:street" type of keys to a dictionary like this:
{"address": {"street": "Some value"}}
So, we have to see if we have such tags, and if we have any tags with problematic characters.

Please complete the function 'key_type', such that we have a count of each of four tag categories in a dictionary:
  "lower", for tags that contain only lowercase letters and are valid,
  "lower_colon", for otherwise valid tags with a colon in their names,
  "problemchars", for tags with problematic characters, and
  "other", for other tags that do not fall into the other three categories.
"""
OSMFILE = "perth_australia_sample.osm"
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def key_type(element, keys):
    if element.tag == "tag":
        k = element.attrib['k']
        if problemchars.search(k):
            keys['problemchars'] += 1
        elif lower_colon.search(k):
            keys['lower_colon'] += 1
        elif lower.search(k):
            keys['lower'] += 1
        else:
            keys['other'] += 1
    return keys



def process_map(osmfile):
    osm_file = open(osmfile, "r")
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(osm_file):
        keys = key_type(element, keys)
    osm_file.close()
    return keys

process_map(OSMFILE)


# ### Explore users

# Below is a set of how many unique users have contributed to the map in this particular area:

# In[8]:

# Exploring Users from the problem set in the course

"""
Your task is to explore the data a bit more.
The first task is a fun one - find out how many unique users
have contributed to the map in this particular area!

The function process_map should return a set of unique user IDs ("uid")
"""
OSMFILE = "perth_australia_sample.osm"
def process_map(osmfile):
    osm_file = open(osmfile, "r")
    users = set()
    for _, element in ET.iterparse(osm_file):
        tag = element.tag
        if tag in [ 'node', 'way', 'relation']:

            id = element.attrib['uid']
            users.add(id)
        pass
    osm_file.close()
    return users

users = process_map(OSMFILE)
pprint.pprint(users)


# Now that we have the data ready, we will start to audit and correct some of the mistakes identified.

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


# ### Audit and correct postcodes

# Next I'll look into the postcodes: Perth postcodes are 4 digit-numbers starting by 6. Using similar functions than the section on streetnames, we can look for unusual postcodes as printed below:

# In[12]:

# # Create a group of auditing functions for postal codes
def audit_postcode(post_code, digits):
    """ Checks if postal code is incompatible and adds it to the list if so. """
    if len(digits) != 4 or digits[0] != '6':
        post_code.append(digits)


def is_postalcode(elem):
    """ Returns a Boolean value."""
    return (elem.attrib['k'] == "addr:postcode")


def audit(osmfile):
    """ Iterates and returns list of inconsistent postal codes found in the document. """
    osm_file = open(osmfile, "r")
    post_code = []
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_postalcode(tag):
                    audit_postcode(post_code, tag.attrib['v'])
    osm_file.close()
    return post_code

# Run audit and print results
postal_codes = audit(OSMFILE)
print postal_codes


# Here WA stands for Western Australia. While those are correct postcodes, we can remove the letters and spaces to harmonize their format with the rest of the dataset. After a similar update process than for the street names, the unusual postcodes are corrected as per below:

# In[13]:

def update_zipcode(post_code):    
    if post_code[0:2] == 'WA' or post_code[0:2] == 'Wa' or post_code[0:2] == 'wa':
        post_code = post_code[3:].strip()
    return post_code

for code in postal_codes:
    better_code = update_zipcode(code)
    print code, "=>", better_code


# ### Convert XLM to CSV

# Now we can transform the data into CSV files ...

# In[14]:

# Preparing for Database from the problem set in the course

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema


# In[15]:

# Check if input element is a "node" or a "way" then clean, shape and parse to corresponding dictionary.

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements
    
    if element.tag == 'node':
        for field in node_attr_fields:
            node_attribs[field] = element.attrib[field]
    
    if element.tag == 'way':
        for field in way_attr_fields:
            way_attribs[field] = element.attrib[field]
        
        position = 0
        temp = {}
        for tag in element.iter("nd"):
            temp['id'] = element.attrib["id"]
            temp['node_id'] = tag.attrib["ref"]
            temp['position'] = position
            position += 1
            way_nodes.append(temp.copy())

    temp = {}
    for tag in element.iter("tag"):
        temp['id'] = element.attrib["id"]
        if ":" in tag.attrib["k"]:
            newKey = re.split(":",tag.attrib["k"],1)
            temp['key'] = newKey[1]
            if temp['key'] == 'postcode':
                temp['value'] = update_zipcode(tag.attrib["v"])
            elif temp['key'] == 'street':
                temp['value'] = update_name(tag.attrib["v"],mapping)
            else:
                temp['value'] = tag.attrib["v"]
            temp["type"] = newKey[0]
        else:
            temp['key'] = tag.attrib["k"]
            if temp['key'] == 'postcode':
                temp['value'] = update_zipcode(tag.attrib["v"])
            elif temp['key'] == 'street':
                temp['value'] = update_name(tag.attrib["v"],mapping)
            else:
                temp['value'] = tag.attrib["v"]
            temp["type"] = default_tag_type
        tags.append(temp.copy())  
        
    
    if element.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# In[16]:

# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_strings = (
            "{0}: {1}".format(k, v if isinstance(v, str) else ", ".join(v))
            for k, v in errors.iteritems()
        )
        raise cerberus.ValidationError(
            message_string.format(field, "\n".join(error_strings))
        )

class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# In[17]:

# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file,          codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file,          codecs.open(WAYS_PATH, 'w') as ways_file,          codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file,          codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


# In[18]:

process_map(OSMFILE, validate=False)


# ## Analyze the data with SQL

# ### Import CSV into SQL

# ... and import those CSV files into an SQLite3 Database

# In[19]:

import sqlite3
db = sqlite3.connect("PerthWA.db")
c = db.cursor()


# In[20]:

c.execute('''DROP TABLE IF EXISTS nodes''')
db.commit()


# In[21]:

c.execute('''
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY NOT NULL,
    lat REAL,
    lon REAL,
    user TEXT,
    uid INTEGER,
    version INTEGER,
    changeset INTEGER,
    timestamp TEXT
);
''')

c.execute('''
CREATE TABLE nodes_tags (
    id INTEGER,
    key TEXT,
    value TEXT,
    type TEXT,
    FOREIGN KEY (id) REFERENCES nodes(id)
);
''')

c.execute('''
CREATE TABLE ways (
    id INTEGER PRIMARY KEY NOT NULL,
    user TEXT,
    uid INTEGER,
    version TEXT,
    changeset INTEGER,
    timestamp TEXT
);
''')

c.execute('''
CREATE TABLE ways_tags (
    id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT,
    FOREIGN KEY (id) REFERENCES ways(id)
);
''')

c.execute('''
CREATE TABLE ways_nodes (
    id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY (id) REFERENCES ways(id),
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);
''')

db.commit()


# In[22]:

# Read in the csv file as a dictionary, format the data as a list of tuples:
with open('nodes.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['lat'], i['lon'], i['user'].decode("utf-8"), i['uid'], i['version'], i['changeset'], i['timestamp']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO nodes(id, lat, lon, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", to_db)
# commit the changes
db.commit()


# In[23]:

with open('nodes_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['key'], i['value'].decode("utf-8"), i['type']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO nodes_tags(id, key, value,type) VALUES (?, ?, ?, ?);", to_db)
# commit the changes
db.commit()


# In[24]:

with open('ways.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['user'].decode("utf-8"), i['uid'], i['version'], i['changeset'], i['timestamp']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO ways(id, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?);", to_db)
# commit the changes
db.commit()


# In[25]:

with open('ways_nodes.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['node_id'], i['position']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO ways_nodes(id, node_id, position) VALUES (?, ?, ?);", to_db)
# commit the changes
db.commit()


# In[26]:

with open('ways_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['key'], i['value'].decode("utf-8"), i['type']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO ways_tags(id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
# commit the changes
db.commit()


# ### Queries

# Files sizes:

# In[27]:

# Getting the list of files and their size
dirpath = '.'

files_list = []
for path, dirs, files in os.walk(dirpath):
    files_list.extend([(filename, size(os.path.getsize(os.path.join(path, filename)))) for filename in files])

for filename, size in files_list:
    print '{:.<40s}: {:5s}'.format(filename,size)


# Count of ways:

# In[28]:

# Count of ways
query = "SELECT COUNT(*) FROM ways;"
c.execute(query)
c.fetchall()[0][0]


# Count of nodes:

# In[29]:

# Count of nodes
query = "SELECT COUNT(*) FROM nodes;"
c.execute(query)
c.fetchall()[0][0]


# Top 5 contributing users and their number of contributions:

# In[30]:

# Top 5 contributing users and their number of contributions
query = "SELECT e.user, COUNT(*) as num FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) e GROUP BY e.user ORDER BY num DESC LIMIT 5;"
c.execute(query)
c.fetchall()


# 10 most popular fast food chains and their count of restaurants:

# In[31]:

# 10 most popular fast food chains and their count of restaurants
query = "SELECT nodes_tags.value, COUNT(*) as num FROM nodes_tags JOIN (SELECT DISTINCT(id) FROM nodes_tags WHERE value='fast_food') as r     ON nodes_tags.id=r.id WHERE nodes_tags.key='name' GROUP BY nodes_tags.value ORDER BY num DESC LIMIT 10;"
c.execute(query)
c.fetchall()


# 10 most represented types of amenities and their occurence count:

# In[32]:

# 10 most represented types of amenities and their occurence count
query = "SELECT value, COUNT(*) as num FROM nodes_tags WHERE key='amenity' GROUP BY value ORDER BY num DESC LIMIT 15;"
c.execute(query)
c.fetchall()


# ## Conclusion

# In this project I selected and downloaded a metro extract of the geographical data related to the city of Perth, Australia. After sampling it to end up with a smaller OSM file, I audited it by looking at users, streetnames, and postcodes, and created functions to clean some of the errors linked to that sample file. Finally, I converted that OSM file into separate CSV files for nodes, tags and ways, and loaded those files into SQL to perform some exploratory queries over the dataset.

# Overall I found the source data to be quite clean and standardized already, with only a few abbreviations used for street names and a couple of postcodes wrongly formatted. Nevertheless, it is still achievable to improve the auditing process of OpenStreetMap.

# ### Improvement ideas

# - Input Standardization: one of the first ideas would be to standardize the data from the input phase, by setting up rules to avoid wrong data entries. For instance, we saw in the postcode exercise that postcodes for Perth have 4 digits, and can either be written 'WA 0000' or simply '0000'. 
# 
# Benefit: Making a final decision about the format by automatically removing any letters from the postcodes inputs and flagging the ones that have more or less than 4 digits in a further audit will make sure the wrong inputs have a higher probability to get picked up in a later audit.
# 
# Anticipated problem: Depending of the size of the area investigated, the list of anomalies could be quite time-consuming to audit, and it will be difficult to ensure that the result is coherent without a precise mapping of localities and postcodes.

# - Leveraging timestamps: Similarly, focusing on the timestamp associated with the inputs could improve the auditing process, should this audit be carried regularly and over a long enough timeline: the most recent entries can be audited more thorougly than the older ones which have already been cleaned.
# 
# Benefit: if audits are performed regularly and thoroughly, focusing on the new additions can save time.
# 
# Anticipated problem: if then correcting the mistakes in the recent inputs updates the timestamps, the corrected inputs will then appear again in the next review, adding an unecessary workload.

# - Users' contributions: Investigate further the users' contributions can also give us interesting insights into the data: from the "Explore Users" section, we ranked the top contributors to the Perth dataset and their count of submissions. What if we also kept track of the proportion of errors or anomalies flagged for all users? 
# 
# Benefit: We could then have a better understanding of which users submit the error-prone contributions, and what submissions should be looked into in priority.
# 
# Anticipated problem: this analysis will have to be kept running on an ongoing basis, accounting for the additions and removals of users. This can prove to be difficult to set up or maintain.

# - Third party tools (such as the Google Map API) could also be used to improve our exisiting data.
# 
# Benefit: this will leverage existing data.
# 
# Anticipated problem: this will be difficult to put into place at the beginning, as the two datasets could differ vastly. How do we review the different versions of the same node with accuracy and efficiency?
