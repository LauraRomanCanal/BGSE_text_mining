'''plots'''

cc_y.plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))
#barplot
#df = cc_y.head(5)
#df.plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="bar").legend(loc='center left', bbox_to_anchor=(1, 0.5))

#let's locate peaks of each series
cc_y.sort(columns='pos',axis=0, ascending=False)[:3] #1815
cc_y.sort(columns='neg',axis=0, ascending=False)[:3] #1908
cc_y.sort(columns='unc',axis=0, ascending=False)[:3] #1805
cc_y.sort(columns='passive',axis=0, ascending=False)[:3] #1822
cc_y.sort(columns='ethic',axis=0, ascending=False)[:3] #1811
cc_y.sort(columns='polit',axis=0, ascending=False)[:3] #1964
cc_y.sort(columns='econ',axis=0, ascending=False)[:3] #1949
cc_y.sort(columns='milit',axis=0, ascending=False)[:3] #1814

peak_dates = [1815,1908, 1805, 1811, 1964,1949,1814]
peak_text = [4,5,1,2,7,6,3]
#what happened in the years?
#1) UNCERTAINTY: 1805 end of First Barbary War, #end of 1802-1804 recession
#2) ETHICS: 1811: Slave revolt in Louisiana, – Battle of Tippecanoe: American troops led by William Henry Harrison defeat the Native American chief Tecumseh.
#3) MILITAR: 1814: Anglo-American war 1812-1815
#4) POSITIVE 1815: Treaty of Ghent (end of Anglo-American war)
#5) NEGATIVE 1908:  Panic of 1907, the fallout from the panic led to Congress creating the Federal Reserve System
#6) ECONOMY: 1949: Recession of 1949
#7) POLITICS: 1964: Legislation in the U.S. Congress on Civil Rights is passed. It banned discrimination in jobs, voting and accommodations. The Tonkin Resolution is passed by the United States Congress, authorizing broad powers to the president to take action in Vietnam after North Vietnamese boats had attacked two United States destroyers five days earlier.
#https://en.wikipedia.org/wiki/List_of_recessions_in_the_United_States
#http://americasbesthistory.com/abhtimeline1940.html

    ''' plot peakes'''  
X = cc_y['year']
Y1 = cc_y['pos'];Y2= cc_y['neg'];Y3 = cc_y['unc'];Y4= cc_y['passive']
Y5 = cc_y['ethic'];Y6= cc_y['polit'];Y7 = cc_y['econ']; Y8= cc_y['milit']

#key dates in US history
us_dates = [1812,1861,1865,1868,1898,1917,1929,1941,1945, 1974,1991,2001,2008]
us_dates_exp = ['War on Britain','Civil War','Lincoln is shot',
                'US citizens equal rights','Maine explosion',
                'WWI','Black Thursday','WWII','WWII ends',
                'Watergate scandal','Iraq attacks','WTC attack', 'Great Recession']


#peaks
plt.plot(X, Y3,   lw = 1.)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('%')
plt.title('Uncertainty on speeches', y=1.08)
plt.axvline(1805,color="black",linestyle='dashed',lw=0.5)
plt.text(1805,1,'1805 - End Bavary War and Recession',rotation=0)
plt.savefig('uncert.png', bbox_inches='tight')


plt.plot(X, Y5,   lw = 1.)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('%')
plt.title('Ethics on speeches', y=1.08)
plt.axvline(1811,color="black",linestyle='dashed',lw=0.5)
plt.text(1811,4.5,'1811 -  Battle of Tippecanoe and slave revolts',rotation=0)
plt.savefig('ethics.png', bbox_inches='tight')

plt.plot(X, Y8,   lw = 1.)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('%')
plt.title('Military on speeches', y=1.08)
plt.axvline(1812,color="black",linestyle='dashed',lw=0.5)
plt.axvline(1815,color="black",linestyle='dashed',lw=0.5)
plt.text(1814,1,'1812-1815 - Anglo-American war ',rotation=0)
plt.savefig('milit.png', bbox_inches='tight')

plt.plot(X, Y1,   lw = 1.)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('%')
plt.title('Positivity on speeches', y=1.08)
plt.axvline(1815,color="black",linestyle='dashed',lw=0.5)
plt.text(1815,36,'1815 - Treaty of Ghent (end Anglo-American war) ',rotation=0)
plt.savefig('posit.png', bbox_inches='tight')

plt.plot(X, Y2,   lw = 1.)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('%')
plt.title('Negativity on speeches', y=1.08)
plt.axvline(1908,color="black",linestyle='dashed',lw=0.5)
plt.text(1908,12,'1908 - Panic of 1907  ',rotation=0)
plt.savefig('negat.png', bbox_inches='tight')


plt.plot(X, Y7,   lw = 1.)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('%')
plt.title('Economy on speeches', y=1.08)
plt.axvline(1949,color="black",linestyle='dashed',lw=0.5)
plt.text(1928,12,'1949 - Recession of 1949 ',rotation=0)
plt.savefig('econ.png', bbox_inches='tight')


plt.plot(X, Y6,   lw = 1.)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('%')
plt.title('Politics on speeches', y=1.08)
plt.axvline(1964,color="black",linestyle='dashed',lw=0.5)
plt.text(1934,12,'1964 -  Tonkin Resolution ',rotation=0)
plt.savefig('politc.png', bbox_inches='tight')
