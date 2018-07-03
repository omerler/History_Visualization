~ Other look at the History ~
Visualization project by 
	Omer Arie Lerinman, Omerler@gmail.com
Guided by
	Ph.D. Michael Fink
At the 
	Hebrew university of Jerusalem

~ The goal ~
To enable intuitive historical exploration, by those who seek for interesting, 
yet non-conventional patterns and connections between historical events.

~ The problem ~
Statistical history research is a complicated:
	How would you define event?
	Both numerical and textual (parsing)
	size

~ The method ~
Finding historical events similarity in an irregular aspects: all but the chronological and geographical features

~ Data collection – The SSP public ~
The “SSP public” (Societal Stability Protocol) is a technology-intensive effort to extract event data from a global archive of news reports covering the Post WWII era.
Initiator:  “Cline” institute, University of Illinois. 
ALL DATASET RIGHTS ARE RESERVED TO THE CLINE INSTITUTE (Huge thanks!)
Size: Over 52,000 worldwide sociological-oriented events, each one with above 106 fields (dimensions) which describe it’s geographical, chronical and demographical characteristics. 

~ Dataset limitations ~
The major limitation of this Dataset is the lack of informative string label, i.e. the common name of the event (“world war two”, for example). 
Two minor additional limitations, is the non-standardized input vector for each event and the unbalanced geographical events’ division.

~ Data processing ~
The following actions were made:
	Duplications removal
	Geographical and chronological (but month) attributes were removed
	Numerical and binary dimensions were normalized, where non-exist numerical attributes were replaced with the column average
	Enum dimensions were replaces with indicator dimensions (“Dummy values”)
	A PCA, implemented by scikit.learn package, output a 2D variant of the data

~ Visualisation approach ~
In order to stand this mission, I created a neat interactive GUI which on one hand present all data at once but on the other hand, enable the user to “dive into” 
interesting singular events for further research and exploration.
The two dimensions holds the main variance between the different events, while the coloring method selected 
represent chosen feature and the length of the event represented by it’s size. 

~ Implementation details ~
Data processing was written in python using :
	Scipy, Pandas & Scikit packages for data processing
	Matplotlib.pyplot for GUI

~ How to? ~
1. Clone directory
2. Validate needed packeges
3. Run 'Visualisation.py' file

Enjoy!