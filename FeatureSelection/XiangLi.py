
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import multivariate_normal




# In[3]:

def monte_carlo_simulation(n, confidence_value=0.99):
    
    #sample_size = 100
    alpha = 1 - confidence_value
    low = ([])
    high = ([])
    xRange = ([])
    
    for j in range(n):
        sample_size = math.pow(10,j)

        y = ([])
        #density probability function
        for i in np.arange(sample_size):

            x1 = np.random.uniform(0.5,1)
            x2 = np.random.uniform(0.75,2)
            y.append(multivariate_normal([1,1],[[1,0.3],[0.3,1]]).pdf([x1,x2]))
          
        y = pd.DataFrame(y)
        h = t.ppf(1 - alpha/2,sample_size-1) * (y.std()/np.sqrt(sample_size));
        high.append((0.5*1.25)*(y.mean()+h));
        low.append((0.5*1.25)*(y.mean()-h));
        xRange.append(math.pow(10,j));
    plt.figure(1)
    plt.plot(xRange, high, label = 'High Boundry')
    plt.plot(xRange, low, label = 'Low Boundry')
    #plt.axis([0.0,5, 0.13,0.16])
    plt.legend()
    plt.title("Probability of Multivariate Distribution")
    plt.show()
    
    return low,high
    


# In[4]:

_=monte_carlo_simulation(5)


# In[5]:

df = pd.read_csv('wdbc.data.csv')


# In[6]:

base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                 'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names


# In[7]:

def p_value_equality_normal_means(str):
    
    malig = df[df['class'] == 'M'][str]
    benign = df[df['class'] == 'B'][str]
    
    mean1 = malig.mean()
    mean2 = benign.mean()    
    n = malig.count()
    m = benign.count()
    std1 = malig.std()
    std2 = benign.std()

    
    s = ((n-1.0)*std1**2.0+(m-1.0)*std2**2.0)/(n+m-2.0)
    T = (mean1 - mean2)/(np.sqrt(s*(1.0/n+1.0/m)))
    
    p_value = t(n+m-2.0).cdf(T);

    p_value = min(p_value, 1.0-p_value)
    p_value *= 2.0
    
    return [T,p_value]


# In[8]:

testStatValue = []
name = []
for i in df.columns:
    if((i != 'id' )& (i != 'class')):
        stat,p = p_value_equality_normal_means(i)
        testStatValue.append(stat)
        name.append(i)
df1 = pd.DataFrame(testStatValue) 
df1 = df1.set_index([name])

df1.columns = ['Statistic_Values']
df1 = df1.sort(['Statistic_Values'],ascending=[0])
df1 = df1[0:10]

df1.plot(kind = 'bar')


# In[9]:

def combined_p_value_equality_normal_means(feature1, feature2):
    
    c = 1    
    W = df[feature1] + c*df[feature2]
    Wm = df[df['class'] == 'M'][feature1]/df[feature1].std() + c*df[df['class'] == 'M'][feature2]/df[feature2].std()
    Wb = df[df['class'] == 'B'][feature1]/df[feature1].std() + c*df[df['class'] == 'B'][feature2]/df[feature2].std()    
    m = Wm.count()
    n = Wb.count()
    var = 1+c**2+ 2* c* np.cov(df[feature1]/df[feature1].std(), df[feature2]/df[feature2].std())[0,1]    
    meanWm = Wm.mean()
    meanWb = Wb.mean()   
    T = abs(meanWm - meanWb)/((np.sqrt((1/n+1/m)*var)))
    maxTS = T
    maxc = c
    
    c = -1    
    W = df[feature1] + c*df[feature2]
    Wm = df[df['class'] == 'M'][feature1]/df[feature1].std() + c*df[df['class'] == 'M'][feature2]/df[feature2].std()
    Wb = df[df['class'] == 'B'][feature1]/df[feature1].std() + c*df[df['class'] == 'B'][feature2]/df[feature2].std()    
    m = Wm.count()
    n = Wb.count()
    var = 1+c**2+ 2* c* np.cov(df[feature1]/df[feature1].std(), df[feature2]/df[feature2].std())[0,1]    
    meanWm = Wm.mean()
    meanWb = Wb.mean()   
    T = abs(meanWm - meanWb)/((np.sqrt((1/n+1/m)*var)))
    if(maxTS < T):
        maxTS = T
        maxc = c
    
    delta1 = (df[df['class'] == 'M'][feature1]/df[feature1].std()).mean() - (df[df['class'] == 'B'][feature1]/df[feature1].std()).mean()
    delta2 = (df[df['class'] == 'M'][feature2]/df[feature2].std()).mean() - (df[df['class'] == 'B'][feature2]/df[feature2].std()).mean()
    rho = np.cov(df[feature1]/df[feature1].std(), df[feature2]/df[feature2].std())[0,1] 
    c = (delta2 - delta1*rho)/(delta1 - delta2 * rho)
    
    W = df[feature1] + c*df[feature2]
    Wm = df[df['class'] == 'M'][feature1]/df[feature1].std() + c*df[df['class'] == 'M'][feature2]/df[feature2].std()
    Wb = df[df['class'] == 'B'][feature1]/df[feature1].std() + c*df[df['class'] == 'B'][feature2]/df[feature2].std()    
    m = Wm.count()
    n = Wb.count()
    var = 1+c**2+ 2* c* np.cov(df[feature1]/df[feature1].std(), df[feature2]/df[feature2].std())[0,1]    
    meanWm = Wm.mean()
    meanWb = Wb.mean()   
    T = abs(meanWm - meanWb)/((np.sqrt((1/n+1/m)*var)))
    if(maxTS < T):
        maxTS = T
        maxc = c
    
    return maxTS, maxc


# In[10]:

Mts = 0
Mc = 0
for i in df.columns:
    for j in df.columns:
        if((i != 'id' )& (i != 'class') & (j != 'id' )& (j != 'class') & (i!=j)):
            ts, c = combined_p_value_equality_normal_means(feature1 = i, feature2 = j)
            if(ts > Mts):
                Mts = ts
                Mc = c
                best1 = i
                best2 = j
a1 = 1/df[best1].std() 
a2 = Mc / df[best2].std() 
print Mts,"From",a1,best1,"and",a2,best2


# In[11]:

Y = a1 * df[best1] + a2 * df[best2]
Ym = a1*df[df['class'] == 'M'][best1] + a2*df[df['class'] == 'M'][best2]
Yb = a1*df[df['class'] == 'B'][best1] + a2*df[df['class'] == 'B'][best2]

plt.figure(3)
ax = Ym.plot(kind = "hist",legend=True)
ax = Yb.plot(kind = "hist",legend=True)
l = plt.legend(('Malignant lumps','Benign lumps.'), loc='best')
plt.show()





