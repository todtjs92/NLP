# LDA 깁스샘플링


## 1.data 

#### 1_1 : ap.dat
dat 형식으로 각 문서별 단어가 index로 표현 되있음.
ex) I love you -> 3 1 2 
#### 1_2 :vocab.txt
문서에 있는 모든 단어 사전.

## 2. LDAGibbs.py
### Gibbs_Sampling에 필요한 모듈
#### 다음과 같은 구조로 작동

#### __init__
![init](https://user-images.githubusercontent.com/59354220/118360824-d3b47000-b5c3-11eb-952a-393316af6a05.jpeg)

#### _gibbs_sampling(self)

![gipps](https://user-images.githubusercontent.com/59354220/118360870-170ede80-b5c4-11eb-910d-674d75d6abc0.jpeg)



## RUN_LDAVI.py 
#### 실행 python3 RUN_LDAVI.py
