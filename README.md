# Pinckley_Python_Portfolio
Code I wrote for my BISC 450C python class

## Jupyter Notebooks

```python
%matplotlib inline 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")
```


```python
df = pd.read_csv("/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25495</td>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <td>25496</td>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <td>25497</td>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <td>25498</td>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <td>25499</td>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>228</td>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>290</td>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>294</td>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>296</td>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>352</td>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(df.profit[non_numeric_profits])
```




    {'N.A.'}




```python
len(df.profit[non_numeric_profits])
```




    369




```python
bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins = range(1955, 2006))
```





```python
![output_11_0](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/24cf3ded-6e3a-4ff4-ac55-0a64d06743a5)

```



## Analyzing Data 

```python
import numpy
```


```python
numpy.loadtxt(fname ="swc-python/data/inflammation-01.csv", delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
data = numpy.loadtxt(fname ="swc-python/data/inflammation-01.csv", delimiter = ',')
```


```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
print(type(data))
```

    <class 'numpy.ndarray'>



```python
print(data.shape)
```

    (60, 40)



```python
print("first value in data:", data[0,0])
```

    first value in data: 0.0



```python
print("middle value in data:", data[29,19])
```

    middle value in data: 16.0



```python
print(data[0:4, 0:10])
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10,0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data[:3,36:]
```


```python
print("small is:\n",small)
```

    small is:
     [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
# lets use a numpy function
print(numpy.mean(data))
```

    6.14875



```python
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)

print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566



```python
#maxval = numpy.amax(data)
#minval = numpy.amin(data)
#stdval = numpy.std(data)
```


```python
print("Maximum inflammation:", maxval)
print("minimum inflammation:", minval)
print("Standard deviation:", stdval)
```

    Maximum inflammation: 20.0
    minimum inflammation: 0.0
    Standard deviation: 4.613833197118566



```python
#look at statistical values
patient_0 = data[0,:] # 0 on 1st axis, everything on the second
print("Maximum inflammation for patient 0:", numpy.amax(patient_0))
```

    Maximum inflammation for patient 0: 18.0



```python
print("Maximum inflammation for patient 2:", numpy.amax(data[2,:]))
```

    Maximum inflammation for patient 2: 19.0



```python
print(numpy.mean(data,axis = 0))
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
print(numpy.mean(data,axis = 0).shape)
```

    (40,)



```python
print(numpy.mean(data, axis = 1))
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]



```python

```

## Visualizing Data

```python
import numpy
data = numpy.loadtxt(fname = "inflammation-01.csv", delimiter = ',')
```


```python
#heatmap of inflammation over time
import matplotlib.pyplot
image = matplotlib.pyplot.imshow(data)
matplotlib.pyplot.show()
```


![output_1_0](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/31c56b12-2cf7-4972-a985-ad90062ff69c)




```python
#Average inflamtion over time
ave_inflammation = numpy.mean(data,axis=0)
ave_plot = matplotlib.pyplot.plot(ave_inflammation)
matplotlib.pyplot.show
```




    <function matplotlib.pyplot.show(*args, **kw)>





![output_2_1](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/eddb4803-6916-4cbe-86fc-02fd1da085d7)



```python
max_plot = matplotlib.pyplot.plot(numpy.amax(data, axis=0))
matplotlib.pyplot.show
```




    <function matplotlib.pyplot.show(*args, **kw)>






![output_3_1](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/c5d964d1-a79e-4218-8634-ff72a89d0c8e)


```python
min_plot = matplotlib.pyplot.plot(numpy.amin(data, axis=0))
matplotlib.pyplot.show
```




    <function matplotlib.pyplot.show(*args, **kw)>





![output_4_1](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/e20ff727-cee5-482f-a4c7-5f5db25b3d7b)



```python
fig = matplotlib.pyplot.figure(figsize =(10.0,3.0))

axes1 = fig.add_subplot(1, 3, 1)
axes2 = fig.add_subplot(1, 3, 2)
axes3 = fig.add_subplot(1, 3, 3)

axes1.set_ylabel("average")
axes1.plot(numpy.mean(data, axis =0))

axes2.set_ylabel("max")
axes2.plot(numpy.amax(data,axis=0))

axes3.set_ylabel("min")
axes3.plot(numpy.amin(data,axis=0))

fig.tight_layout()

matplotlib.pyplot.savefig("inflammation.png")
matplotlib.pyplot.show()
```



![output_5_0](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/716a56d0-6bef-4121-b6a4-6a5b287239e4)



```python

```

## Storing Values in Lists



```python
odds = [1,3,5,7]
print("odds are:", odds)
```

    odds are: [1, 3, 5, 7]



```python
print("First element:", odds[0])
print("last element:", odds[3])
print("-1 element:", odds[-1])
```

    First element: 1
    last element: 7
    -1 element: 7



```python
names = ["Curie", "Darwing", "Turing"] #Typo in Darwin's name

print("names are originally:", names)

names[1] = "Darwin" #Correct name

print("Final value of names:", names)
```

    names are originally: ['Curie', 'Darwing', 'Turing']
    Final value of names: ['Curie', 'Darwin', 'Turing']



```python
#name = "Darwin"
#name[0]
```


```python
odds.append(11)
print("odds after adding a value:", odds)
```

    odds after adding a value: [1, 3, 5, 7, 11]



```python
removed_element = odds.pop(0)
print("Odds after removing the first element:", odds)
print("Removed_element:", removed_element)
```

    Odds after removing the first element: [3, 5, 7, 11]
    Removed_element: 1



```python
odds.reverse()
print("odds after reversing:", odds)
```

    odds after reversing: [11, 7, 5, 3]



```python
odds = [3,5,7]
primes = odds
primes.append(2)
print("Primes:", primes)
print("Odds:", odds)
```

    Primes: [3, 5, 7, 2]
    Odds: [3, 5, 7, 2]



```python
odds = [3,5,7]
primes = list(odds)
primes.append(2)
print("Primes:", primes)
print("Odds:", odds)
```

    Primes: [3, 5, 7, 2]
    Odds: [3, 5, 7]



```python
binomial_name = "Drosophila melanogaster"
group = binomial_name[0:10]
print("group:", group)

species = binomial_name[11:23]
print("species:", species)

chromosomes = ["X","Y","2","3","4"]
autosomes = chromosomes[2:5]
print("Autosomes:", autosomes)

last = chromosomes[-1]
print("last:", last)
```

    group: Drosophila
    species: melanogaster
    Autosomes: ['2', '3', '4']
    last: 4



```python
date = "Monday 4 January 2023"
day = date[0:6]
print("Using 0 to begin range:", day)
day = date[:6]
print("Omitting beginning index:", day)
```

    Using 0 to begin range: Monday
    Omitting beginning index: Monday



```python
months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
sond = months[8:12]
print("With known last position:", sond)

sond = months[8:len(months)]
print("Using len() to get last entry:", sond)

sond = months[8:]
print("Omitting ending index", sond)
```

    With known last position: ['sep', 'oct', 'nov', 'dec']
    Using len() to get last entry: ['sep', 'oct', 'nov', 'dec']
    Omitting ending index ['sep', 'oct', 'nov', 'dec']



```python

```



## Using Loops


```python
odds = [1,3,5,7]
```


```python
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5
    7



```python
odds = [1,3,5]
print(odds[0])
print(odds[1])
print(odds[2])
```

    1
    3
    5



```python
odds = [1,3,5,7,9,11,13,15,17,19]

for num in odds:
    print(num)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19



```python
length = 0
names = ["Curie", "Darwin", "Turing"]
for value in names:
    length = length + 1
print("There are",length,"names in the list.")
```

    There are 3 names in the list.



```python
name = "Rosalind"
for name in ["Curie","Darwin","Turing"]:
    print(name)
print("after the loop, name is", name)
```

    Curie
    Darwin
    Turing
    after the loop, name is Turing



```python
print(len([0,1,2,3]))
```

    4



```python
name = ["Curie", "Darwin", "Turing"]

print(len(name))
```

    3



```python

```


## Using Multiple Files


```python
import glob
import numpy
import matplotlib.pyplot
```


```python
print(glob.glob("inflammation*.csv"))
```

    ['inflammation-05.csv', 'inflammation-12.csv', 'inflammation-04.csv', 'inflammation-08.csv', 'inflammation-10.csv', 'inflammation-06.csv', 'inflammation-09.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-11.csv', 'inflammation-03.csv', 'inflammation-02.csv']



```python
filenames = sorted(glob.glob("inflammation*.csv"))
filenames = filenames[0:3]

for filename in filenames:
    print(filename)
    
    data = numpy.loadtxt(fname = filename, delimiter=",")
    
    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
    
    axes1 = fig.add_subplot(1,3,1)
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel("average")
    axes1.plot(numpy.mean(data, axis =0))

    axes2.set_ylabel("max")
    axes2.plot(numpy.amax(data,axis=0))

    axes3.set_ylabel("min")
    axes3.plot(numpy.amin(data,axis=0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    

```

    inflammation-01.csv

![output_2_1](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/2833b49d-90fa-4550-b9f4-2bc6dbe85e34)




    inflammation-02.csv


![output_2_3](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/e3399b4d-de41-43c6-885a-34425ee7f38b)



    inflammation-03.csv



![output_2_5](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/81027fb8-a27f-42c5-b254-0358bd3a78af)





```python

```


## Making Choices pt.1

```python
num = 37
if num > 100:
    print("Greater")
else:
    print("not greater")
print("done")
```

    not greater
    done



```python
num = 53
print("before conditional...")
if num > 100:
    print(num, "is greater than 100")
print("...After Conditional")

```

    before conditional...
    ...After Conditional



```python
num = -3

if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, "is 0")
else:
    print(num, "is negative")
```

    -3 is negative



```python
if (1>0) and (-1 >= 0):
    print("Both parts are true")
else:
    print("At least one part is false")
```

    At least one part is false



```python
if (1>0) or (-1 >= 0):
    print("At least one part is true")
else:
    print("Both are false")
```

    At least one part is true



```python
import numpy
```


```python

```


## Making Choices pt. 2

```python
import numpy
```


```python
data = numpy.loadtxt(fname = "inflammation-01.csv", delimiter = ',')
```


```python
max_inflammation_0 = numpy.amax(data, axis = 0)[0]

```


```python
max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print("Suspicious looking maxima!")
elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print("Minima add up to zero!")
else:
    print("Seems OK!")
```

    Suspicious looking maxima!



```python
data = numpy.loadtxt(fname = "inflammation-03.csv", delimiter = ',')

max_inflammation_0 = numpy.amax(data, axis = 0)[0]

max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print("Suspicious looking maxima!")
elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print("Minima add up to zero! -> Healthy Person Alert!!")
else:
    print("Seems OK!")
```

    Minima add up to zero! -> Healthy Person Alert!!



```python

```



## Functions pt.1

```python
fahrenheit_val = 99
celcius_val = ((fahrenheit_val - 32)*(5/9))
print(celcius_val)
```

    37.22222222222222



```python
fahrenheit_val2 = 43
celcius_val2 = ((fahrenheit_val2 - 32)*(5/9))
print(celcius_val2)
```

    6.111111111111112



```python
def explicit_fahr_to_celsius(temp):
    # Assign the converted value to a variable
    converted = ((temp - 32)*(5/9))
    # Return value of new value
    return converted
```


```python
def fahr_to_celsius(temp):
    # Return conberted values more effeciently using return func without using new variable. Does the same thing, but more streamlined
    return ((temp - 32)*(5/9))
```


```python
fahr_to_celsius(32)
```




    0.0




```python
explicit_fahr_to_celsius(32)
```




    0.0




```python
print("Freexing point of water:", fahr_to_celsius(32), "C")
print("Boiling point of water:", fahr_to_celsius(212), 'C')
```

    Freexing point of water: 0.0 C
    Boiling point of water: 100.0 C



```python
def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

print("Freezing point of water in Kelvin:", celsius_to_kelvin(0.))
```

    Freezing point of water in Kelvin: 273.15



```python
def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k

print("Boiling point of water:", fahr_to_kelvin(212), 'F')

```

    Boiling point of water: 373.15 F



```python
def fahr_to_k(temp_f):
    return celsius_to_kelvin(fahr_to_celsius(temp_f))

print("Boiling point of water:", fahr_to_k(212), 'F')
```

    Boiling point of water: 373.15 F



```python
temp_kelvin = fahr_to_kelvin(212.0)
print("Temp in Kelvin was:", temp_kelvin)
```

    Temp in Kelvin was: 373.15



```python
def print_temperatures():
    print("Temperature in Fahrenheit was:", temp_fahr)
    print("Temperature in Kelvin was:", temp_kelvin)
    
temp_fahr = 212.0
temp_kelvin = fahr_to_kelvin(temp_fahr)

print_temperatures()
```

    Temperature in Fahrenheit was: 212.0
    Temperature in Kelvin was: 373.15



```python

```


## Functions pt.2-4

```python
import numpy
import matplotlib.pyplot
import glob
```


```python


def visualize(filename):
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    fig = matplotlib.pyplot.figure(figsize=(10.0,3.0))
    
    axes1 = fig.add_subplot(1,3,1)
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel("Average")
    axes1.plot(numpy.mean(data, axis=0))
    
    axes2.set_ylabel("max")
    axes2.plot(numpy.amax(data,axis=0))

    axes3.set_ylabel("min")
    axes3.plot(numpy.amin(data,axis=0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```


```python
def detect_problems(filename):
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    max_inflammation_0 = numpy.amax(data, axis = 0)[0]
    max_inflammation_20 = numpy.amax(data, axis = 0)[20]
    
    if max_inflammation_0 == 0 and max_inflammation_20 == 20:
        print("Suspicious looking maxima!")
    elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
        print("Minima add up to zero!")
    else:
        print("Seems OK!")
    
```


```python
filenames = sorted(glob.glob("inflammation*.csv"))

for filename in filenames:
    print(filename)
    visualize(filename)
    detect_problems(filename)
```

    inflammation-01.csv

![output_3_1](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/a37cc6ff-a858-4699-951d-31bd8af50ea1)





    Suspicious looking maxima!
    inflammation-02.csv


![output_3_3](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/cfc47645-8585-471d-9a0d-51693a816d2a)



    Suspicious looking maxima!
    inflammation-03.csv


![output_3_5](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/1fb06e59-3448-49f1-b3ee-c43365edec9f)




    Minima add up to zero!
    inflammation-04.csv


![output_3_7](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/05def088-82d4-4e9b-abb6-39bda4415033)




    Suspicious looking maxima!
    inflammation-05.csv


![output_3_9](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/ec8d05c2-d556-491f-a13f-3b13fde88702)




    Suspicious looking maxima!
    inflammation-06.csv


![output_3_11](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/269aa97b-ab7a-4edb-8b8b-8c1126203e29)




    Suspicious looking maxima!
    inflammation-07.csv


![output_3_13](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/699887bc-9839-4149-8d53-c5873d7343b3)



    Suspicious looking maxima!
    inflammation-08.csv


![output_3_15](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/ec3e0e18-3bca-4985-89f1-c13f6e65e7c4)




    Minima add up to zero!
    inflammation-09.csv


![output_3_17](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/a224d7d6-e82e-4fcc-93d6-4f1b6a1bfaf9)




    Suspicious looking maxima!
    inflammation-10.csv


![output_3_19](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/746ae7a7-0bd0-45ff-8b13-c89cb69f031a)




    Suspicious looking maxima!
    inflammation-11.csv


![output_3_21](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/f1c165f1-1422-4cc5-b9fd-474c4aa1e499)




    Minima add up to zero!
    inflammation-12.csv


![output_3_23](https://github.com/Laurel-Pinckley/Pinckley_Python_Portfolio/assets/134239816/359c637e-5d86-44cb-bedf-a5aa2381d125)




    Suspicious looking maxima!



```python
def offset_mean(data, target_mean_value):
    return(data - numpy.mean(data)) + target_mean_value
```


```python
z = numpy.zeros((2,2))
print(offset_mean(z,3))
```

    [[3. 3.]
     [3. 3.]]



```python
data = numpy.loadtxt(fname = "inflammation-01.csv", delimiter=',')

print(offset_mean(data,0))
```

    [[-6.14875 -6.14875 -5.14875 ... -3.14875 -6.14875 -6.14875]
     [-6.14875 -5.14875 -4.14875 ... -5.14875 -6.14875 -5.14875]
     [-6.14875 -5.14875 -5.14875 ... -4.14875 -5.14875 -5.14875]
     ...
     [-6.14875 -5.14875 -5.14875 ... -5.14875 -5.14875 -5.14875]
     [-6.14875 -6.14875 -6.14875 ... -6.14875 -4.14875 -6.14875]
     [-6.14875 -6.14875 -5.14875 ... -5.14875 -5.14875 -6.14875]]



```python
print("origninal min, mean, and max are:", numpy.amin(data), numpy.mean(data), numpy.amax(data))
offset_data = offset_mean(data,0)
print("offset min, mean, and max are:", numpy.amin(offset_data), numpy.mean(offset_data), numpy.amax(offset_data))
```

    origninal min, mean, and max are: 0.0 6.14875 20.0
    offset min, mean, and max are: -6.14875 2.842170943040401e-16 13.85125



```python
print("std dev before and after:", numpy.std(data),numpy.std(offset_data))
```

    std dev before and after: 4.613833197118566 4.613833197118566



```python
print("difference in standard deviation before and after:", numpy.std(data) - numpy.std(offset_data))
```

    difference in standard deviation before and after: 0.0



```python
# offset_meant(data, target_mean_value):
#return a new array conataining the orig data with its mean offset to match the desired value
def offset_mean(data, target_mean_value):
    return(data - numpy.mean(data)) + target_mean_value

```


```python
def offset_mean(data, target_mean_value):
    """Return a new array containing the original data with its mean offset to match the desired value"""
    return(data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data with its mean offset to match the desired value
    



```python
def offset_mean(data, target_mean_value):
    """Return a new array containing the original data
    with its ean offset to match the desired value.
    
    Examples
    -------------
    
    >>> Offset_mean([1,2,3],0)
    array([-1., 0., 1.])
    """
    return(data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data
        with its ean offset to match the desired value.
        
        Examples
        -------------
        
        >>> Offset_mean([1,2,3],0)
        array([-1., 0., 1.])
    



```python
numpy.loadtxt("inflammation-01.csv", delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
def offset_mean(data, target_mean_value = 0.0):
    """Return a new array containing the original data
    with its ean offset to match the desired value, (0 by default).
    
    Examples
    -------------
    
    >>> Offset_mean([1,2,3])
    array([-1., 0., 1.])
    """
    return(data - numpy.mean(data)) + target_mean_value
```


```python
test_data = numpy.zeros((2,2))
print(offset_mean(test_data,3))
```

    [[3. 3.]
     [3. 3.]]



```python
print(offset_mean(test_data))
```

    [[0. 0.]
     [0. 0.]]



```python
def display(a=1,b=2,c=3):
    print("a:",a,"b:",b,"c:",c)

print("no parameters:")
display()
print("one parameter:")
display(55)
print("two parameters:")
display(55,66)
```

    no parameters:
    a: 1 b: 2 c: 3
    one parameter:
    a: 55 b: 2 c: 3
    two parameters:
    a: 55 b: 66 c: 3



```python
print("only setting the value of c")
display(c=77)
```

    only setting the value of c
    a: 1 b: 2 c: 77



```python
help(numpy.loadtxt)
```

    Help on function loadtxt in module numpy:
    
    loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
        Load data from a text file.
        
        Each row in the text file must have the same number of values.
        
        Parameters
        ----------
        fname : file, str, or pathlib.Path
            File, filename, or generator to read.  If the filename extension is
            ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings for Python 3k.
        dtype : data-type, optional
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        comments : str or sequence of str, optional
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter : str, optional
            The string used to separate values. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will parse the
            column string into the desired value.  E.g., if column 0 is a date
            string: ``converters = {0: datestr2num}``.  Converters can also be
            used to provide a default value for missing data (but see also
            `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines, including comments; default: 0.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.
        
            .. versionchanged:: 1.11.0
                When a single column has to be read it is possible to use
                an integer instead of a tuple. E.g ``usecols = 3`` reads the
                fourth column the same way as ``usecols = (3,)`` would.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
            data-type, arrays are returned for each field.  Default is False.
        ndmin : int, optional
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed.
            Legal values: 0 (default), 1 or 2.
        
            .. versionadded:: 1.6.0
        encoding : str, optional
            Encoding used to decode the inputfile. Does not apply to input streams.
            The special value 'bytes' enables backward compatibility workarounds
            that ensures you receive byte arrays as results if possible and passes
            'latin1' encoded strings to converters. Override this value to receive
            unicode arrays and pass strings as input to converters.  If set to None
            the system default is used. The default value is 'bytes'.
        
            .. versionadded:: 1.14.0
        max_rows : int, optional
            Read `max_rows` lines of content after `skiprows` lines. The default
            is to read all the lines.
        
            .. versionadded:: 1.16.0
        
        Returns
        -------
        out : ndarray
            Data read from the text file.
        
        See Also
        --------
        load, fromstring, fromregex
        genfromtxt : Load data with missing values handled as specified.
        scipy.io.loadmat : reads MATLAB data files
        
        Notes
        -----
        This function aims to be a fast reader for simply formatted files.  The
        `genfromtxt` function provides more sophisticated handling of, e.g.,
        lines with missing values.
        
        .. versionadded:: 1.10.0
        
        The strings produced by the Python float.hex method can be used as
        input for floats.
        
        Examples
        --------
        >>> from io import StringIO   # StringIO behaves like a file object
        >>> c = StringIO(u"0 1\n2 3")
        >>> np.loadtxt(c)
        array([[0., 1.],
               [2., 3.]])
        
        >>> d = StringIO(u"M 21 72\nF 35 58")
        >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
        ...                      'formats': ('S1', 'i4', 'f4')})
        array([(b'M', 21, 72.), (b'F', 35, 58.)],
              dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])
        
        >>> c = StringIO(u"1,0,2\n3,0,4")
        >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
        >>> x
        array([1., 3.])
        >>> y
        array([2., 4.])
    



```python
numpy.loadtxt("inflammation-01.csv", delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python

```

## Defensive Programming


```python
numbers = [1.5,2.3,0.7,0.001,4.4]
total = 0.0
for num in numbers:
    assert num> 0.0, "Data should only contain positive values"
    total += num
print("Total is:", total)
```

    Total is: 8.901



```python
def normalize_rectangle(rect):
    """Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    input should be of the format (x0, y0, x1, y1).
    (x0, y0) and (x1, y1) define the lower left and upper right corners of the rectangle respectively."""
    assert len(rect) == 4, "Rectangles must contain 4 coordinates"
    x0, y0, x1, y1 = rect
    assert x0 < x1, "Invalid X coordinates"
    assert y0 < y1, "Invalid Y coordinates"
    
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy:
        scaled = dy/dx
        upper_x, upper_y = 1.0, scaled
    else:
        scaled = dx/dy
        upper_x, upper_y = scaled, 1.0
        
    assert 0 < upper_x <= 1.0, "Calculated upper X coordinate invalid"
    assert 0 < upper_y <= 1.0, "Calculated upper Y coordinate invalid"
    
    return(0, 0, upper_x, upper_y)
```


```python
#print(normalize_rectangle((0.0, 1.0, 2.0)))
```


```python
#print(normalize_rectangle((4.0, 1.0, 2.0, 5.0)))
```


```python
print(normalize_rectangle((0.0, 0.0, 1.0, 5.0)))
```

    (0, 0, 0.2, 1.0)



```python
print(normalize_rectangle((0.0, 0.0, 5.0, 1.0)))
```

    (0, 0, 1.0, 0.2)



```python

```


## Transcribing DNA to RNA

```python
# Prompt user to enter inupt file name

input_file_name = input("Enter the name of the input fasta file")
```

    Enter the name of the input fasta file lambda_int.txt



```python
# Open input file and read sequence
with open(input_file_name, "r") as input_file:
    dna_sequence = ""
    for line in input_file:
        if line.startswith(">"):
            continue
        dna_sequence += line.strip()
    
```


```python
# Transcribing DNA to RNA
rna_sequence = ""
for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide
```


```python
#prompt user to enter output file name

output_file_name = input("Enter the name of the output file: ")
```

    Enter the name of the output file:  lambda_int_RNA.txt



```python
#print sequence to text file

with open(output_file_name, "w") as output_file:
    output_file.write(rna_sequence)
    print("The RNA sequence has been saved to {output_file_name}")
```

    The RNA sequence has been saved to {output_file_name}



```python
print(rna_sequence)
```

    AUGGGAAGAAGGCGAAGUCAUGAGCGCCGGGAUUUACCCCCUAACCUUUAUAUAAGAAACAAUGGAUAUUACUGCUACAGGGACCCAAGGACGGGUAAAGAGUUUGGAUUAGGCAGAGACAGGCGAAUCGCAAUCACUGAAGCUAUACAGGCCAACAUUGAGUUAUUUUCAGGACACAAACACAAGCCUCUGACAGCGAGAAUCAACAGUGAUAAUUCCGUUACGUUACAUUCAUGGCUUGAUCGCUACGAAAAAAUCCUGGCCAGCAGAGGAAUCAAGCAGAAGACACUCAUAAAUUACAUGAGCAAAAUUAAAGCAAUAAGGAGGGGUCUGCCUGAUGCUCCACUUGAAGACAUCACCACAAAAGAAAUUGCGGCAAUGCUCAAUGGAUACAUAGACGAGGGCAAGGCGGCGUCAGCCAAGUUAAUCAGAUCAACACUGAGCGAUGCAUUCCGAGAGGCAAUAGCUGAAGGCCAUAUAACAACAAACCAUGUCGCUGCCACUCGCGCAGCAAAAUCAGAGGUAAGGAGAUCAAGACUUACGGCUGACGAAUACCUGAAAAUUUAUCAAGCAGCAGAAUCAUCACCAUGUUGGCUCAGACUUGCAAUGGAACUGGCUGUUGUUACCGGGCAACGAGUUGGUGAUUUAUGCGAAAUGAAGUGGUCUGAUAUCGUAGAUGGAUAUCUUUAUGUCGAGCAAAGCAAAACAGGCGUAAAAAUUGCCAUCCCAACAGCAUUGCAUAUUGAUGCUCUCGGAAUAUCAAUGAAGGAAACACUUGAUAAAUGCAAAGAGAUUCUUGGCGGAGAAACCAUAAUUGCAUCUACUCGUCGCGAACCGCUUUCAUCCGGCACAGUAUCAAGGUAUUUUAUGCGCGCACGAAAAGCAUCAGGUCUUUCCUUCGAAGGGGAUCCGCCUACCUUUCACGAGUUGCGCAGUUUGUCUGCAAGACUCUAUGAGAAGCAGAUAAGCGAUAAGUUUGCUCAACAUCUUCUCGGGCAUAAGUCGGACACCAUGGCAUCACAGUAUCGUGAUGACAGAGGCAGGGAGUGGGACAAAAUUGAAAUCAAAUAA



```python

```



## Translation of RNA to DNA


```python
# Prompt user to enter the input file name

input_file_name = input("Enter the name of the input RNA file")
```

    Enter the name of the input RNA file lambda_int_RNA.txt



```python
# Open and read RNA file

with open(input_file_name, "r") as input_file:
    rna_sequence = input_file.read().strip()
```


```python
# Define the codon table
codon_table = { 
    'UCA': 'S',    
    'UCC': 'S',    
    'UCG': 'S',    
    'UCU': 'S',    
    'UUC': 'F',    
    'UUU': 'F',   
    'UUA': 'L',   
    'UUG': 'L',   
    'UAC': 'Y',   
    'UAU': 'Y',   
    'UAA': '*',   
    'UAG': '*',   
    'UGC': 'C',   
    'UGU': 'C',    
    'UGA': '*',    
    'UGG': 'W',    
    'CUA': 'L',   
    'CUC': 'L',    
    'CUG': 'L',    
    'CUU': 'L',    
    'CCA': 'P',    
    'CCC': 'P',   
    'CCG': 'P',    
    'CCU': 'P',    
    'CAC': 'H',    
    'CAU': 'H',    
    'CAA': 'Q',    
    'CAG': 'Q',    
    'CGA': 'R',    
    'CGC': 'R',    
    'CGG': 'R',   
    'CGU': 'R',    
    'AUA': 'I',    
    'AUC': 'I',   
    'AUU': 'I',   
    'AUG': 'M',    
    'ACA': 'T',    
    'ACC': 'T',    
    'ACG': 'T',    
    'ACU': 'T',    
    'AAC': 'N',    
    'AAU': 'N',    
    'AAA': 'K',    
    'AAG': 'K',    
    'AGC': 'S',    
    'AGU': 'S',    
    'AGA': 'R',    
    'AGG': 'R',    
    'GUA': 'V',    
    'GUC': 'V',    
    'GUG': 'V',    
    'GUU': 'V',    
    'GCA': 'A',    
    'GCC': 'A',    
    'GCG': 'A',    
    'GCU': 'A',    
    'GAC': 'D',   
    'GAU': 'D',   
    'GAA': 'E',   
    'GAG': 'E',    
    'GGA': 'G',    
    'GGC': 'G',    
    'GGG': 'G',    
    'GGU': 'G'
        }
```


```python
# Translate RNA to protein

protein_sequence = ""
for i in range(0, len(rna_sequence),3):
    codon = rna_sequence[i:i+3]
    if len(codon) == 3:
        amino_acid = codon_table[codon]
        if amino_acid == "*":
            break
        protein_sequence += amino_acid
```


```python
# Prompt the user to enter the output file name

output_file_name = input("Enter the name of the output file: ")
```

    Enter the name of the output file:  lambda_int_protein.txt



```python
# Save the protein sequence to the text file

with open(output_file_name, "w") as output_file:
    output_file.write(protein_sequence)
    print(f"The protein sequence has been saved to {output_file_name}")
```

    The protein sequence has been saved to lambda_int_protein.txt



```python
print(protein_sequence)
```

    MGRRRSHERRDLPPNLYIRNNGYYCYRDPRTGKEFGLGRDRRIAITEAIQANIELFSGHKHKPLTARINSDNSVTLHSWLDRYEKILASRGIKQKTLINYMSKIKAIRRGLPDAPLEDITTKEIAAMLNGYIDEGKAASAKLIRSTLSDAFREAIAEGHITTNHVAATRAAKSEVRRSRLTADEYLKIYQAAESSPCWLRLAMELAVVTGQRVGDLCEMKWSDIVDGYLYVEQSKTGVKIAIPTALHIDALGISMKETLDKCKEILGGETIIASTRREPLSSGTVSRYFMRARKASGLSFEGDPPTFHELRSLSARLYEKQISDKFAQHLLGHKSDTMASQYRDDRGREWDKIEIK



```python

```
