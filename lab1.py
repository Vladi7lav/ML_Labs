# импортирование библиотек
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as pt

df = pd.read_csv(r'titanic/train.csv')# чтение данных

print(df.head()) # отображение первых пяти строк
print(df.tail()) # отображение последних пяти строк
print(df.info()) # информация о данных
df.describe() # отображение статистики по каждому числовому признаку
df_changes = df.dropna()  # удаление строк или столбцов, не содержащих данные или имеющих пропуски в данных
df_changes = df_changes.drop_duplicates() # удаление строк, содержащих одинаковые данные
print(df_changes.info())
print(df.shape)  #  отображение размера данных
print("##################################################")

#1. Какое количество мужчин и женщин ехало на корабле?
ns1 = []
ns1.extend(df.Sex)
print ('male: ' + str(ns1.count('male')))
print ('female: ' + str(ns1.count('female')))
print("##################################################")

#2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
ns2 = []
ns2.extend(df.Survived)
lucky2 = round(ns2.count(1)/len(ns2),2)
print ('lucky: ' + str(lucky2) + ' or ' + str(lucky2*100) + '% (used rounding to hundredths)')
print("##################################################")

#3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
np = []
np.extend(df.Pclass)
lucky3 = round(np.count(1)/len(np),2)
print ('lucky: ' + str(lucky3) + ' or ' + str(lucky3*100) + '% (used rounding to hundredths)')
print("##################################################")

#4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. 
print ('The average age of the passengers: ' + str(round(df['Age'].mean(),2)) + ', median: ' + str(df['Age'].median()))
print("##################################################")

#5. Коррелируют ли число братьев/сестер с числом родителей/детей?
#  Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
correlation =  df["SibSp"].corr(df["Parch"])
print ('Correlation: ' + str(round(correlation,2)))
print("##################################################")

#6. Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name). 
#Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
# а также разделения их на женские и мужские.

# Выбераем из таблицы  строки только с женскими именами и создадём последовательность из столбца 'Name'
pop_name = (df[df["Sex"] == "female"])["Name"].reset_index(drop=True)

#выкидываем строки, идущие до запятой и точки
pop_name = pop_name.apply(lambda x: x.split(',')[1].split('.')[1])

#тоже самое, для открывающей скобки
for i, name in enumerate(pop_name):
    if name.find("(") != -1:
        pop_name[i] = name.split("(")[1]

pop_name = pop_name.apply(lambda x: x.strip(' ').split(' ')[0])

names_frequency = pop_name.value_counts()

print("Most popular name is: " + str(names_frequency.idxmax()) + ".  It is repeated" + str(names_frequency.max()))
print("##################################################")

#7. Коррелирует ли класс, которым ехал пассажир, с выживаемостью?
correlation = df['Pclass'].corr(df["Survived"])

print ('Correlation: ' + str(round(correlation,2)))
print("##################################################")

#8. Визуализируйте гистограммы возраста для выживших и не выживших пассажиров. Сделайте выводы. 
# Отобразите данные на одном и нескольких графиках

df_Tsurv = df[df['Survived'] == True]
df_Fsurv = df[df['Survived'] == False]

df_Tsurv_ages = df_Tsurv['Age']
df_Fsurv_ages = df_Fsurv['Age']

#Подготавливаем данные для построения графиков
df_Tsurv_ages = df_Tsurv_ages.dropna().reset_index(drop=True)
df_Fsurv_ages = df_Fsurv_ages.dropna().reset_index(drop=True)

sns.distplot(df_Fsurv_ages, bins=10, hist=True, label='Not survived', kde=False)
sns.distplot(df_Tsurv_ages, bins=10, hist=True, label='Survived', kde=False)

pt.legend()
pt.show()
print("##################################################")

#9. Визуализируйте гистограммы возраста для выживших и не выживших пассажиров по классам. Сделайте выводы. 
for cls in range(1, 4):
    title = str(cls) + ' сlass' 
    
    for surv in range(2):
        if surv == True:
            event = 'Survived'
        else:
            event = "Not survived"
            
        df_MRW = df[df['Survived'] == surv]         #MRW: money rules the world
        df_MRW = df_MRW[df_MRW['Pclass'] == cls]
        df_MRW = df_MRW['Age']
        
        df_MRW = df_MRW.dropna().reset_index(drop=True)
        sns.distplot(df_MRW, bins=10, hist=True, label=event, kde=False)
        
    pt.title(title)
    pt.legend()
    pt.show()
print("##################################################")

#10. Постройте столбчатую диаграмму количества людей: мужчины, женщины, дети.
df_children = df[df['Age'] < 18]
df_no_longer_children = df[df['Age'] >= 18]
df_children = df_children.replace(to_replace=['male', 'female'], value=['kid', 'kid'])

df_new = pd.concat([df_children, df_no_longer_children])

sns.countplot(x='Sex', data = df_new)
pt.show()
print("##################################################")