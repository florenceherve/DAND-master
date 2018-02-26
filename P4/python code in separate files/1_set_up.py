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
