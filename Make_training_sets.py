#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:24:27 2024

@author: mattc

So far, all my training sets have been species specific.

This script is to combine all 24 species and make easily-digestible training sets 
that consist of equal mixes of all species

each mixed training set is a length of chunk size
"""

import pandas as pd

filepath = "/home/mattc/Documents/Fun/Training_data_EC_predictor/species_sequences/"
species = {"mouse":"mmusculus_sequences.xlsx", "intest": "cintestinalis_sequences.xlsx", "corn":"zmays_sequences.xlsx", "tgondii":"tgondii_sequences.xlsx", "yeast":"scerevisiae_sequences.xlsx", "celegans":"celegans_sequences.xlsx", "zebrafish":"drerio_sequences.xlsx", "athaliana":"athaliana_sequences.xlsx", "ecoli": "ecoli_sequences.xlsx", "pfalc":"pfalciparum_sequences.xlsx", "cbot":"cbotulinum_sequences.xlsx", "aspergillus": "anidulans_sequences.xlsx", "algae":"creinhardtii_sequences.xlsx", "fly":"dmelanogaster_sequences.xlsx", "chicken":"ggallus_sequences.xlsx", "mold":"ncrassa_sequences.xlsx", "rice":"osativa_sequences.xlsx", "baboon":"panubis_sequences.xlsx", "saureus":"saureus_sequences.xlsx", "afulgidus":"afulgidus_sequences.xlsx", "bsubtilis":"bsubtilis_sequences.xlsx", "ddiscoideum":"ddiscoideum_sequences.xlsx", "glamblia":"glamblia_sequences.xlsx", "macetivorans":"macetivorans_sequences.xlsx"}

chunk_size = 10000   
max_gene_length = 2000
    
def ceil(x):
    return int(x+1) - int(1-(x-int(x)))


#this block of code actually combines all training data into one massive dataframe and mixes it
frame = 0
for x in species:
    if type(frame) != pd.core.frame.DataFrame:
        frame = pd.read_excel(filepath+species[x], sheet_name = "Sheet0")
        frame = frame[frame["Length"] <= max_gene_length]
    else:
        temp = pd.read_excel(filepath+species[x], sheet_name = "Sheet0")
        temp = temp[temp["Length"] <= max_gene_length]
        frame = pd.concat([frame, temp])
        del temp
del x
#this randomizes the row order
frame = frame.sample(frac=1).reset_index(drop=True)
    

#this outputs the dataframe into files of the desired chunk size
for x in range(ceil(len(frame)/chunk_size)):
    if x == (ceil(len(frame)/chunk_size)-1):
        chunk = frame.iloc[(x*chunk_size)::,:]
    else:
        chunk = frame.iloc[(x*chunk_size):((x+1)*chunk_size),:]
    chunk.to_excel("/home/mattc/Documents/Fun/Training_data_EC_predictor/mixed_TD2000_"+str(x+1)+".xlsx", sheet_name = "Sheet0", index = False)
    del chunk  
del x       
    