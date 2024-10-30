import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pi=open("pi.rtf").read()
pi=pi[0]+pi[2:-1]
x=np.array(range(len(pi))).reshape(-1,1)
y=[d in "01234" for d in pi]
m=0
for i in range(1_000_000):
    clf=RandomForestClassifier(n_estimators=1,max_depth=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.9999)
    clf.fit(x_train,y_train)
    m+=clf.score(x_test,y_test)
    print(i,m/(i+1))

