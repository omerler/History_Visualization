#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ================================================ General Info ========================================================

"""
Course: Visualization and Sonification
Name: Omer Arie Lerinman
"""

# ===================================================== Imports ========================================================
import pandas as                        pd
import numpy as                         np
import matplotlib.pyplot as             plt
import matplotlib.colors
from matplotlib.widgets import          Button

from sklearn.decomposition import       PCA
from sklearn.cluster import             KMeans
import                                  pickle
from mpldatacursor import               datacursor
from matplotlib import                  cm, colors as Colors
from scipy.spatial import               Voronoi, voronoi_plot_2d
from scipy.misc import                  imread

import seaborn as                       sns

# scalable color bar
import mynormalize
from draggable_colorbar import DraggableColorbar
# =============================================== Design ==================================================


# Set style of plots
sns.set_style('white')

# Define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)

font = {'family': 'serif',
        # 'color':  'darkred',
        'weight': 'normal',
        'size': 10
        }
matplotlib.rc('font', **font)

# =============================================== Dataset description ==================================================

loc_type = {0:"Whole Country",
    1:'Sparsely Populated Area',
    2:'Densely Populated Area',
    3:'Border Area',
    4:'Road',
	5:'Checkpoint',
	6:'Bridge',
	7:'Railway',
	8:'Harbor',
	9:'Residential Property',
	10:'Shopping Area',
	11:'Hotel',
	12:'Ground Transportation Center',
	13:'Airport',
	14:'Industrial Property',
	15:'Office Complex',
	16:'Religious Site',
	17:'Medical Facility',
	18:'School Site',
	19:'Recreational Site',
	20:'Public Monument',
	21:'Government Facility',
	22:'Military Facility',
	23:'Embassy/Consulate',
	24:'Air Space',
	25:'Water Space',
	26:'Rebel Stronghold',
	27:'Unspecified Location',
	28:'Other Location'}

e_length = {1:'One Day',
	2:'Two to Seven Days',
	3:'One Week to a Month',
	4:'Longer than a Month',
}
ev_type	= {0:"Missing",
	1:"Political Expression",
	2:"Political Attacks",
	# 3: None,
	4:"Disruptive State Acts",
	5:"Political Reconfigurations",
	# 9:"Other",
}

region ={1:"SS Africa",
	2:"Asia",
	3:"Europe",
	4:"Latin America & Caribbean",
	5:"North America",
	6:"Oceania",
	7:"Northern Africa",
	8:"Middle East",
}

weapon ={1:"No Weapon Used",
	2:"Fake weapon Used",
	3:"Body Parts",
	4:"Animal",
	5:"Vehicles",
	6:"Computer",
	7:"Blunt Instrument",
	8:"Tear gas, Mace, etc.",
	9:"Knives/sharp Instrument",
	10:"improvised explosive device",
	11:"Letter Bomb",
	12:"Fire",
	13:"Non-lethal Projectiles",
	14:"Small Arms",
	15:"Light Weapon",
	16:"Incendiary Device",
	17:"Land Mine",
	18:"Explosives Grenade",
	19:"Car Bomb",
	20:"Tanks/armored Vehicles",
	21:"Field Artillery",
	22:"Missile/rocket",
	23:"Aircraft Munitions",
	24:"Naval Power",
	25:"Biochemical Weapons"
}


# =============================================== Dataset labels ==================================================
DAY = 'DAY'
COUNTRY = 'COUNTRY'
YEAR = 'YEAR'
MONTH = 'MONTH'
EVENT_TYPE = 'EV_TYPE'

irrelevant_labels = ['EVENTID', 'COWCODE', 'CODE_YEAR', 'CODE_MONTH', 'CODE_DAY', 'JUL_PSD',
                     'JUL_PED', DAY,COUNTRY,YEAR, 'JUL_EED', 'JUL_LED',
                     'JUL_START_DATE', 'JUL_END_DATE', 'AID', 'PUB_MON', 'PUB_DATE', 'PUB_YEAR', 'RECAP',
                     'EVENT', 'POSTHOC', 'LINKED', 'LINK_TYPE', 'FROM_EID', 'TO_EID', 'PINPOINT', 'GP7', 'GP8']

# Names and regions
REGION = 'REGION'
group_names_involved = ['INI_IGRP1', 'TAR_IGRP1', 'VIC_IGRP1', 'INI_SGRP1', 'TAR_SGRP1', 'VIC_SGRP1', 'INI_PGRP1',
                        'TAR_PGRP1', 'VIC_PGRP1', 'GP3', 'GP4', REGION ]
irrelevant_labels += group_names_involved

# unclear labels
unclear_labels = ['KNOW_INI', 'AMBIG_INI', 'DATE_TYP', 'AD_TACT', 'AMBIG_WGT', 'QUASI_EVENT']
irrelevant_labels += unclear_labels

# labels to transform to dummy-values
enums_columns = ['INI_TYPE', 'NGOV_I1', 'GOV_I1', 'TAR_TYPE', 'HUMAN_T1', 'GOV_T1', 'HUMAN_V1', 'GOV_V1',
                 'VIC_TYPE', 'GP_TYPE', EVENT_TYPE, 'LOC_TYPE', 'PE_TYPE', 'SYM_TYPE', 'EXP_TYPE', 'ATK_TYPE',
                 'DSA_TYPE', 'STAT_ACT', 'WEAP_GRD', 'WEAPON', 'PROPERTY_DAMAGED', 'PROPERTY_OWNER', 'NEWS_SOURCE']

# ===================================================== Constants ======================================================

nonvalues = ['.', np.NaN, None, np.nan]
SHOW_DATA_MODE = False
FILENAME = 'pca_ssp'
CMAP = plt.get_cmap('Spectral')
NUM_CLUSTERS = 9
# ===================================================== Implementation==================================================
       
def isnan(obj, more_values=[]):
    try:
        return np.isnan(obj) or obj in nonvalues+more_values
    except:
        return obj in nonvalues+more_values
        

def ssp_public():
    original_df, df_reduced, pca, new_X = get_data()
    df_dict = original_df.to_dict('index')
   
    x = new_X[:,0]
    y = new_X[:,1]
    
    length = np.array(df_reduced.E_LENGTH) 
    fig = plt.figure(frameon=False)
    fig.suptitle('      The History\n      by Omer Arie Lerinman', fontdict=font)
    ax = fig.add_subplot(111)
    ax.axis('off')

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 5), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"), annotation_clip=False)
    annot.set_visible(False)

    class ChangeColormap(object):
        def __init__(self):
            self._spaces = ""

            # region coloring initiation
            self.region_colormap = list(map(lambda x: float(x), original_df[REGION].tolist()))
            self.regions = list(region.values())
            self.regions_norm = self.norm(self.region_colormap)
            self.regions_colormap = cm.ScalarMappable(self.regions_norm, cmap=plt.get_cmap('rainbow'))  # past: rainbow
            self.regions_colors = self.regions_colormap.to_rgba(self.region_colormap)
            self.regions_legend = []
            box = dict(boxstyle='roundtooth', fc="w", ec="k")

            for i, y_val in enumerate(np.arange(start=0.1, step=0.11, stop=.98)):
                self.regions_legend.append(ax.text(0, y_val, self.regions[i],
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, style='oblique', weight='roman', backgroundcolor='gray', size='large',
                        family='fantasy', color=self.regions_colormap.to_rgba(i + 1), fontsize=11, bbox=box))
                self.regions_legend[i].set_visible(False)

            # event type coloring initiation
            self.event_type_colormap = list(map(lambda x: float(x), original_df[EVENT_TYPE].tolist()))
            self.events_types = list(ev_type.values())
            self.events_types_norm = self.norm(self.event_type_colormap)
            self.events_types_colormap = cm.ScalarMappable(self.events_types_norm, cmap=plt.get_cmap('Set2'))
            self.events_types_colors = self.events_types_colormap.to_rgba(self.event_type_colormap)
            self.events_types_legend = []
            box = dict(boxstyle='roundtooth', fc="w", ec="k")

            for i, y_val in enumerate([0.5,0.6,0.7,0.8,0.9]):
                self.events_types_legend.append(ax.text(0, y_val, self.events_types[i],
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes, style='oblique', weight='roman', backgroundcolor='gray', size='large',
                        family='fantasy', color=self.events_types_colormap.to_rgba(i + 1), fontsize=11, bbox=box))
                self.events_types_legend[i].set_visible(False)

            # year coloring initiation
            self.year = original_df.YEAR
            self.years = list(np.arange(start=1948, step=10, stop=2018))
            self.years.reverse()
            self.year_legend = []
            self._year_norm = self.norm(self.year)
            self._year_colormap = cm.ScalarMappable(self._year_norm, 'gnuplot')
            self._year_colors = self._year_colormap.to_rgba(self.year)
            for i, y_val in enumerate(np.arange(start=0.08, step=0.13, stop=0.99)):
                self.year_legend.append(ax.text(0, y_val, str(self.years[i]), verticalalignment='bottom',
                                                horizontalalignment='left', transform=ax.transAxes, style='oblique',
                                                weight='roman', backgroundcolor='gray', size='large', family='fantasy',
                                                color = self._year_colormap.to_rgba(self.years[i]), fontsize=11,
                                        bbox=box))
                self.year_legend[i].set_visible(True)
            self._year_colormap.set_array(self.year)

        @staticmethod
        def norm(array):
            return Colors.Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
    
        def init_colors(self):

            n_killed_p = list(df_reduced['PER_ATK_I'])
            n_killed_a = list(df_reduced['PER_ATK_I_new'])
            colors = np.maximum(n_killed_p, n_killed_a) * (y ** 2)
            color_norm = self.norm(colors)
            return colors, color_norm

        def turn_off_legends(self, event=None):
            ax.set_title('')
            for leg in self.regions_legend:
                leg.set_visible(False)
            for leg in self.year_legend:
                leg.set_visible(False)
            for leg in self.events_types_legend:
                leg.set_visible(False)
            
        def color_by_region(self, event):
            self.turn_off_legends()
            ax.set_title(self._spaces + "(colors by region)", fontdict=font)
            for leg in self.regions_legend:
                leg.set_visible(True)
            sc.set_color(self.regions_colors)
            plt.show()
            
        def color_by_type(self, event):
            self.turn_off_legends()
            ax.set_title(self._spaces + "(colors by event type)", fontdict=font)
            for leg in self.events_types_legend:
                leg.set_visible(True)
            sc.set_color(self.events_types_colors)
            sc.set_linewidth(1)
            plt.show()
            
        def color_by_year(self, event):
            self.turn_off_legends()
            ax.set_title(self._spaces + "(colors by year)", fontdict=font)
            try:
                sc.set_color(self._year_colors)
                for leg in self.year_legend:
                    leg.set_visible(True)
            except:  # initiation
                return self._year_colors, self._year_norm
            plt.show()

    callback = ChangeColormap()

    def get_label_str(idx):
        row = df_dict[idx]
        ret_val = ''
        day = int(row[DAY])
        month = int(row[MONTH])
        year = int(row[YEAR])
        ret_val += "Date: {}/{}/{}".format(day, month, year)
        num_datafields = 1
        event_type = row['EV_TYPE']
        if not isnan(event_type, more_values=[0, 3, 9]) and event_type in ev_type:
            ret_val += "\n{}".format(ev_type[int(event_type)])
            num_datafields += 1
        length = row['E_LENGTH']
        if not isnan(length):
            ret_val += "\nLength: {}".format(e_length[int(length)])
            num_datafields += 1
        country = row[COUNTRY]
        reg = row[REGION]
        if not isnan(country):
            ret_val += "\nLocation: {}".format(country)
            num_datafields += 1
        elif not isnan(reg):
            ret_val += "\nLocation: {}".format(reg)
            num_datafields += 1
        loc = row['LOC_TYPE']
        if not isnan(loc,[27,28]):
            ret_val += "\nLocation Type: {}".format(loc_type[int(loc)])
            num_datafields += 1
        killed_p = row['N_KILLED_P']
        killed_a = row['N_KILLED_A']
        if not isnan(killed_p) and int(killed_p) != 0:
            ret_val += "\nKilled: {}".format(int(killed_p))
            num_datafields += 1
        elif not isnan(killed_a) and int(killed_a) != 0:
            ret_val += "\nKilled: {}".format(int(killed_a))
            num_datafields += 1
        side_a = row['INI_IGRP1']
        side_b = row['TAR_IGRP1']
        victim = row['VIC_IGRP1']
        socio_a = row['INI_SGRP1']
        socio_b = row['TAR_SGRP1']
        socio_victim = row['VIC_SGRP1']
        politic_a = row['INI_PGRP1']
        politic_b = row['TAR_PGRP1']
        politic_victim = row['VIC_PGRP1']
        if not isnan(side_a) or not isnan(socio_a) or not isnan(politic_a):
            ret_val += "\nInitiator: {}".format(
                    ', '.join([a for a in [politic_a, side_a, socio_a] if not isnan(a)]))
            num_datafields += 1
        if not isnan(side_b) or not isnan(socio_b) or not isnan(politic_b):
            ret_val += "\nTarget: {}".format(', '.join([a for a in [politic_b, side_b, socio_b] if not isnan(a)]))
            num_datafields += 1
        if not isnan(victim):
            ret_val += "\nVictim: {}".format(victim)
            num_datafields += 1
        elif not isnan(socio_victim):
            ret_val += "\nVictim: {}".format(socio_victim)
            num_datafields += 1
        elif not isnan(politic_victim):
            ret_val += "\nVictim: {}".format(politic_victim)
            num_datafields += 1
        weapon_usage = row['WEAPON']
        if not isnan(weapon_usage) and weapon_usage in weapon:
            ret_val += "\nWeapon: {}".format(weapon[int(weapon_usage)])
            num_datafields += 1
        return ret_val

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        idx = ind["ind"][0]
        text = get_label_str(idx)
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.6)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                ind['ind'] = [ind['ind'][0]]
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                


    fig.canvas.mpl_connect("motion_notify_event", hover)
    colors, color_norm = callback.color_by_year(None)
    tmp_length = np.array(length)
    tmp_length = np.multiply(tmp_length, tmp_length)
    sc = plt.scatter(x, y, marker='D', c=colors, s=tmp_length * 20, alpha=0.5, zorder=1,cmap='viridis')

    clean_button_indices = plt.axes([0.15, 0.05, 0.1, 0.075])
    clean_button = Button(clean_button_indices, 'Clean', hovercolor='white')
    clean_button.on_clicked(callback.turn_off_legends)

    color_by_region_button_indices = plt.axes([0.35, 0.05, 0.1, 0.075]) # 4-tuple of floats rect = [left, bottom,
    # width, height]
    color_by_region_button = Button(color_by_region_button_indices, 'Color by\nWorld Region', hovercolor='white')
    color_by_region_button.on_clicked(callback.color_by_region)

    color_by_year_button_indices = plt.axes([0.55, 0.05, 0.1, 0.075])
    color_by_year_button = Button(color_by_year_button_indices, 'Color by\nYear', hovercolor='white')
    color_by_year_button.on_clicked(callback.color_by_year)


    color_by_event_type_indices = plt.axes([0.75, 0.05, 0.1, 0.075])
    event_type_button = Button(color_by_event_type_indices, 'Color by\nEvent type', hovercolor='white')
    event_type_button.on_clicked(callback.color_by_type)

    ax = fig.get_axes()[0]

    plt.show()
    
def get_data():
    original_df = pd.read_csv('ssp_public.csv', sep=",", )
    if SHOW_DATA_MODE:
        print(original_df.describe())  # get statistics about the data

    original_df.drop_duplicates(inplace=True)  # remove duplications
    df_reduced = original_df.copy()
    df_reduced.drop(irrelevant_labels, axis=1, inplace=True)

    # for enum in enums_columns:
    df_reduced.fillna(df_reduced.mean(), inplace=True)
    df_reduced = df_reduced.join(pd.get_dummies(df_reduced, columns=enums_columns, dummy_na=True, drop_first=True, dtype=np.uint64),
                 lsuffix='_new')
    df_reduced = df_reduced.drop(columns=enums_columns)

    df_reduced_max = df_reduced.max()
    indices_max = df_reduced.idxmax()
    if SHOW_DATA_MODE:
        print(df_reduced_max)
        print(indices_max)
    
    # Pre - processing (include normalization and duplications removal
    df_reduced = df_reduced - df_reduced.mean()
    df_reduced = df_reduced / (df_reduced.max() - df_reduced.min())  # nall values will be equal to zero
    
    df_reduced.drop(['INI_TYPE_nan', 'PROPERTY_DAMAGED_nan', 'PROPERTY_OWNER_nan'], axis=1, inplace=True)
    df_reduced = df_reduced.dropna(axis=1)
    print('after dropna shape: ' + str(df_reduced.shape))
    
    std = df_reduced.std(axis=0)

    if SHOW_DATA_MODE:
        print('is there nulls?\t' + str(df_reduced.isnull().values.any()))
        print('How many nulls?\t' + str(df_reduced.isnull().sum().sum()))
        print(df_reduced.describe())
        print(std)

    pca = PCA(n_components=2)
    new_X = pca.fit_transform(df_reduced)
    
    return original_df, df_reduced, pca, new_X


def get_pca():
    """
    Load pca
    """
    try:
        infile = open(FILENAME, 'rb')
        return pickle.load(infile, encoding='bytes')
    except:
        return None

if __name__ == '__main__':
    ssp_public()
