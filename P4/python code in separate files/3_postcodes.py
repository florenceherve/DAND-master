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
