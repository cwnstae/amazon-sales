[Visit to see my portfolio [Link]](https://cwnstae.github.io/data-analytic-portfolio/).
# Amazon Sales EDA
This project analyzes Amazon sales dataset ([link](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)). to uncover key insights and trends. It showcases my skills in data cleaning, exploratory data analysis (EDA), and visualization and using Python (Pandas, Matplotlib, Seaborn).

# Key objectives include:
 - Identifying sales trends and top-performing products
 - Analyzing customer purchasing behavior
 - Providing data-driven recommendations to boost revenue
Through this project, I demonstrate my ability to turn raw data into actionable insights, essential for a data analyst role.

# About Dataset
This dataset is having the data of 1K+ Amazon Product's Ratings and Reviews as per their details listed on the official website of Amazon
## Features
 - product_id - Product ID
 - product_name - Name of the Product
 - category - Category of the Product
 - discounted_price - Discounted Price of the Product
 - actual_price - Actual Price of the Product
 - discount_percentage - Percentage of Discount for the Product
 - rating - Rating of the Product
 - rating_count - Number of people who voted for the Amazon rating
 - about_product - Description about the Product
 - user_id - ID of the user who wrote review for the Product
 - user_name - Name of the user who wrote review for the Product
 - review_id - ID of the user review
 - review_title - Short review
 - review_content - Long review
 - img_link - Image Link of the Product
 - product_link - Official Website Link of the Product

### Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from pathlib import Path
%matplotlib inline
```
### Load data
```python
current_path = str(Path().absolute())
df = pd.read_csv(current_path + "\data - amazon.csv")
df.info()
df
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1465 entries, 0 to 1464
Data columns (total 16 columns):
 #   Column               Non-Null Count  Dtype 
---  ------               --------------  ----- 
 0   product_id           1465 non-null   object
 1   product_name         1465 non-null   object
 2   category             1465 non-null   object
 3   discounted_price     1465 non-null   object
 4   actual_price         1465 non-null   object
 5   discount_percentage  1465 non-null   object
 6   rating               1465 non-null   object
 7   rating_count         1463 non-null   object
 8   about_product        1465 non-null   object
 9   user_id              1465 non-null   object
 10  user_name            1465 non-null   object
 11  review_id            1465 non-null   object
 12  review_title         1465 non-null   object
 13  review_content       1465 non-null   object
 14  img_link             1465 non-null   object
 15  product_link         1465 non-null   object
dtypes: object(16)
memory usage: 183.3+ KB
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>category</th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>about_product</th>
      <th>user_id</th>
      <th>user_name</th>
      <th>review_id</th>
      <th>review_title</th>
      <th>review_content</th>
      <th>img_link</th>
      <th>product_link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B07JW9H4J1</td>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹399</td>
      <td>₹1,099</td>
      <td>64%</td>
      <td>4.2</td>
      <td>24,269</td>
      <td>High Compatibility : Compatible With iPhone 12...</td>
      <td>AG3D6O4STAQKAY2UVGEUV46KN35Q,AHMY5CWJMMK5BJRBB...</td>
      <td>Manav,Adarsh gupta,Sundeep,S.Sayeed Ahmed,jasp...</td>
      <td>R3HXWT0LRP0NMF,R2AJM3LFTLZHFO,R6AQJGUP6P86,R1K...</td>
      <td>Satisfied,Charging is really fast,Value for mo...</td>
      <td>Looks durable Charging is fine tooNo complains...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Wayona-Braided-WN3LG1-Sy...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B098NS6PVG</td>
      <td>Ambrane Unbreakable 60W / 3A Fast Charging 1.5...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹199</td>
      <td>₹349</td>
      <td>43%</td>
      <td>4.0</td>
      <td>43,994</td>
      <td>Compatible with all Type C enabled devices, be...</td>
      <td>AECPFYFQVRUWC3KGNLJIOREFP5LQ,AGYYVPDD7YG7FYNBX...</td>
      <td>ArdKn,Nirbhay kumar,Sagar Viswanathan,Asp,Plac...</td>
      <td>RGIQEG07R9HS2,R1SMWZQ86XIN8U,R2J3Y1WL29GWDE,RY...</td>
      <td>A Good Braided Cable for Your Type C Device,Go...</td>
      <td>I ordered this cable to connect my phone to An...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Ambrane-Unbreakable-Char...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B096MSW6CT</td>
      <td>Sounce Fast Phone Charging Cable &amp; Data Sync U...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹199</td>
      <td>₹1,899</td>
      <td>90%</td>
      <td>3.9</td>
      <td>7,928</td>
      <td>【 Fast Charger&amp; Data Sync】-With built-in safet...</td>
      <td>AGU3BBQ2V2DDAMOAKGFAWDDQ6QHA,AESFLDV2PT363T2AQ...</td>
      <td>Kunal,Himanshu,viswanath,sai niharka,saqib mal...</td>
      <td>R3J3EQQ9TZI5ZJ,R3E7WBGK7ID0KV,RWU79XKQ6I1QF,R2...</td>
      <td>Good speed for earlier versions,Good Product,W...</td>
      <td>Not quite durable and sturdy,https://m.media-a...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Sounce-iPhone-Charging-C...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B08HDJ86NZ</td>
      <td>boAt Deuce USB 300 2 in 1 Type-C &amp; Micro USB S...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹329</td>
      <td>₹699</td>
      <td>53%</td>
      <td>4.2</td>
      <td>94,363</td>
      <td>The boAt Deuce USB 300 2 in 1 cable is compati...</td>
      <td>AEWAZDZZJLQUYVOVGBEUKSLXHQ5A,AG5HTSFRRE6NL3M5S...</td>
      <td>Omkar dhale,JD,HEMALATHA,Ajwadh a.,amar singh ...</td>
      <td>R3EEUZKKK9J36I,R3HJVYCLYOY554,REDECAZ7AMPQC,R1...</td>
      <td>Good product,Good one,Nice,Really nice product...</td>
      <td>Good product,long wire,Charges good,Nice,I bou...</td>
      <td>https://m.media-amazon.com/images/I/41V5FtEWPk...</td>
      <td>https://www.amazon.in/Deuce-300-Resistant-Tang...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B08CF3B7N1</td>
      <td>Portronics Konnect L 1.2M Fast Charging 3A 8 P...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹154</td>
      <td>₹399</td>
      <td>61%</td>
      <td>4.2</td>
      <td>16,905</td>
      <td>[CHARGE &amp; SYNC FUNCTION]- This cable comes wit...</td>
      <td>AE3Q6KSUK5P75D5HFYHCRAOLODSA,AFUGIFH5ZAFXRDSZH...</td>
      <td>rahuls6099,Swasat Borah,Ajay Wadke,Pranali,RVK...</td>
      <td>R1BP4L2HH9TFUP,R16PVJEXKV6QZS,R2UPDB81N66T4P,R...</td>
      <td>As good as original,Decent,Good one for second...</td>
      <td>Bought this instead of original apple, does th...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Portronics-Konnect-POR-1...</td>
    </tr>
  </tbody>
</table>
</div>

# Data Cleaning
```python
df.isnull().sum()
```
```
product_id             0
product_name           0
category               0
discounted_price       0
actual_price           0
discount_percentage    0
rating                 0
rating_count           2
about_product          0
user_id                0
user_name              0
review_id              0
review_title           0
review_content         0
img_link               0
product_link           0
dtype: int64
```


```python
df.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>category</th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>about_product</th>
      <th>user_id</th>
      <th>user_name</th>
      <th>review_id</th>
      <th>review_title</th>
      <th>review_content</th>
      <th>img_link</th>
      <th>product_link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1463</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1351</td>
      <td>1337</td>
      <td>211</td>
      <td>550</td>
      <td>449</td>
      <td>92</td>
      <td>28</td>
      <td>1143</td>
      <td>1293</td>
      <td>1194</td>
      <td>1194</td>
      <td>1194</td>
      <td>1194</td>
      <td>1212</td>
      <td>1412</td>
      <td>1465</td>
    </tr>
    <tr>
      <th>top</th>
      <td>B07JW9H4J1</td>
      <td>Fire-Boltt Ninja Call Pro Plus 1.83" Smart Wat...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹199</td>
      <td>₹999</td>
      <td>50%</td>
      <td>4.1</td>
      <td>9,378</td>
      <td>[CHARGE &amp; SYNC FUNCTION]- This cable comes wit...</td>
      <td>AHIKJUDTVJ4T6DV6IUGFYZ5LXMPA,AE55KTFVNXYFD5FPY...</td>
      <td>$@|\|TO$|-|,Sethu madhav,Akash Thakur,Burger P...</td>
      <td>R3F4T5TRYPTMIG,R3DQIEC603E7AY,R1O4Z15FD40PV5,R...</td>
      <td>Worked on iPhone 7 and didn’t work on XR,Good ...</td>
      <td>I am not big on camera usage, personally. I wa...</td>
      <td>https://m.media-amazon.com/images/I/413sCRKobN...</td>
      <td>https://www.amazon.in/Wayona-Braided-WN3LG1-Sy...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>3</td>
      <td>5</td>
      <td>233</td>
      <td>53</td>
      <td>120</td>
      <td>56</td>
      <td>244</td>
      <td>9</td>
      <td>6</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

All the data is described as a non-numeric column, and I need to change its data type.
```python
df['discounted_price'] = df['discounted_price'].str.replace("₹","")
df['discounted_price'] = df['discounted_price'].str.replace(",",'')
df['discounted_price'] = df['discounted_price'].astype('float64')

df['actual_price'] = df['actual_price'].str.replace("₹",'')
df['actual_price'] = df['actual_price'].str.replace(",",'')
df['actual_price'] = df['actual_price'].astype('float64')

df['discount_percentage']=df['discount_percentage'].str.replace("%",'')
df['discount_percentage'] = df['discount_percentage'].astype('float64')
df['discount_percentage']=df['discount_percentage']/100

df.describe()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1465.000000</td>
      <td>1465.000000</td>
      <td>1465.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3125.310874</td>
      <td>5444.990635</td>
      <td>0.476915</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6944.304394</td>
      <td>10874.826864</td>
      <td>0.216359</td>
    </tr>
    <tr>
      <th>min</th>
      <td>39.000000</td>
      <td>39.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>325.000000</td>
      <td>800.000000</td>
      <td>0.320000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>799.000000</td>
      <td>1650.000000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1999.000000</td>
      <td>4295.000000</td>
      <td>0.630000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77990.000000</td>
      <td>139900.000000</td>
      <td>0.940000</td>
    </tr>
  </tbody>
</table>
</div>

Check non-numeric in `rating` column
```python
non_numeric = df['rating'][df['rating'].astype(str).str.contains(r'[^0-9.]', regex=True, na=False)]
print(non_numeric)
```

```
1279    |
Name: rating, dtype: object
```
There is a way to correct this data by visiting the Amazon link and either replacing or dropping the data. In this case, I chose to drop it.
```python
df = df.drop(1279)
df['rating'] = df['rating'].astype('float64')
df.describe()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1464.000000</td>
      <td>1464.000000</td>
      <td>1464.000000</td>
      <td>1464.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3126.011906</td>
      <td>5447.002923</td>
      <td>0.477131</td>
      <td>4.096585</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6946.625442</td>
      <td>10878.270001</td>
      <td>0.216274</td>
      <td>0.291674</td>
    </tr>
    <tr>
      <th>min</th>
      <td>39.000000</td>
      <td>39.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>323.750000</td>
      <td>800.000000</td>
      <td>0.320000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>799.000000</td>
      <td>1650.000000</td>
      <td>0.500000</td>
      <td>4.100000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1999.000000</td>
      <td>4303.750000</td>
      <td>0.630000</td>
      <td>4.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77990.000000</td>
      <td>139900.000000</td>
      <td>0.940000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




Check what is the 2 missing values
```python
df[df['rating_count'].isnull()]
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>category</th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>about_product</th>
      <th>user_id</th>
      <th>user_name</th>
      <th>review_id</th>
      <th>review_title</th>
      <th>review_content</th>
      <th>img_link</th>
      <th>product_link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>282</th>
      <td>B0B94JPY2N</td>
      <td>Amazon Brand - Solimo 65W Fast Charging Braide...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹199</td>
      <td>₹999</td>
      <td>80%</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>USB C to C Cable: This cable has type C connec...</td>
      <td>AE7CFHY23VAJT2FI4NZKKP6GS2UQ</td>
      <td>Pranav</td>
      <td>RUB7U91HVZ30</td>
      <td>The cable works but is not 65W as advertised</td>
      <td>I have a pd supported car charger and I bought...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Amazon-Brand-Charging-Su...</td>
    </tr>
    <tr>
      <th>324</th>
      <td>B0BQRJ3C47</td>
      <td>REDTECH USB-C to Lightning Cable 3.3FT, [Apple...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹249</td>
      <td>₹999</td>
      <td>75%</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>💎[The Fastest Charge] - This iPhone USB C cabl...</td>
      <td>AGJC5O5H5BBXWUV7WRIEIOOR3TVQ</td>
      <td>Abdul Gafur</td>
      <td>RQXD5SAMMPC6L</td>
      <td>Awesome Product</td>
      <td>Quick delivery.Awesome ProductPacking was good...</td>
      <td>https://m.media-amazon.com/images/I/31-q0xhaTA...</td>
      <td>https://www.amazon.in/REDTECH-Lightning-Certif...</td>
    </tr>
  </tbody>
</table>
</div>

To anoid data loss and keep important information I impute muissing value with median
