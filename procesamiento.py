import html
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve

TOKEN = r'(?u)\b[a-z]{2,}\b'
SEED = 42

def limpiar(texto):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html.unescape(str(texto)))).strip().lower()

def cargar(path):
    raw = pd.read_csv(path, encoding='utf-8')
    if not {'review', 'sentiment'}.issubset(raw.columns):
        raise ValueError('El CSV debe tener las columnas review y sentiment (positive/negative).')
    df = raw[['review', 'sentiment']].dropna().copy()
    df['sentiment'] = df.sentiment.astype(str).str.strip().str.lower()
    if not set(df.sentiment).issubset({'positive','negative'}):
        raise ValueError('sentiment solo admite positive y negative.')
    df['texto'] = df.review.map(limpiar)
    df = df[df.texto.ne('')]
    conflictos = df.groupby('texto').sentiment.nunique()
    df = df[~df.texto.isin(conflictos[conflictos > 1].index)].drop_duplicates('texto').reset_index(drop=True)
    df['y'] = df.sentiment.map({'positive':1,'negative':0})
    if len(df) < 10 or df.y.nunique() < 2 or df.y.value_counts().min() < 5:
        raise ValueError('Se necesitan al menos 5 reseñas distintas por clase después de la limpieza.')
    it, iv = train_test_split(np.arange(len(df)), test_size=.2, stratify=df.y, random_state=SEED)
    return {'df':df, 'it':it, 'iv':iv, 'originales':len(raw)}

def vocabulario(base, frecuencia):
    df,it,iv = base['df'],base['it'],base['iv']
    if frecuencia > len(it):
        raise ValueError('La frecuencia mínima supera la cantidad de reseñas de entrenamiento.')
    vec = CountVectorizer(binary=True, lowercase=False, token_pattern=TOKEN, min_df=int(frecuencia), dtype=np.float32)
    try:
        X = vec.fit_transform(df.texto.iloc[it]).tocsr()
    except ValueError as exc:
        raise ValueError('Ninguna palabra cumple la frecuencia. Reduce el mínimo.') from exc
    V = vec.transform(df.texto.iloc[iv]).tocsr()
    y = df.y.iloc[it].to_numpy()
    pos = np.asarray(X[y==1].sum(axis=0)).ravel().astype(float)
    neg = np.asarray(X[y==0].sum(axis=0)).ravel().astype(float)
    np_,nn = int((y==1).sum()),int((y==0).sum())
    woe = np.log(((pos+.5)/(np_+1))/((neg+.5)/(nn+1)))
    tabla = pd.DataFrame({'palabra':vec.get_feature_names_out(), 'frecuencia':(pos+neg).astype(int),
        'positivas':pos.astype(int),'negativas':neg.astype(int),'woe':woe,
        'presencia_positivas_pct':100*pos/np_, 'presencia_negativas_pct':100*neg/nn})
    return {'vec':vec,'X':X,'V':V,'tabla':tabla,'y':y,'yv':df.y.iloc[iv].to_numpy()}

def ajustar_pca(X):
    n,p = X.shape
    if min(n,p) < 2:
        return None
    A = X.astype(np.float64)
    mu = np.asarray(A.mean(axis=0)).ravel()
    total = float(np.sum(mu*(1-mu))*n/(n-1))
    if total < 1e-14:
        return None
    def producto(v):
        return (np.asarray(A.T@(A@v)).ravel()-n*mu*(mu@v))/(n-1)
    if p <= 3:
        C = np.column_stack([producto(v) for v in np.eye(p)])
        vals, vect = eigh(C)
        vals,vect = vals[-2:],vect[:,-2:]
    else:
        op = LinearOperator((p,p),matvec=producto,dtype=np.float64)
        vals,vect = eigsh(op,k=2,which='LA',v0=np.random.default_rng(SEED).normal(size=p),tol=1e-7,maxiter=max(2000,10*p))
    orden = np.argsort(vals)[::-1]
    componentes = vect[:,orden]
    # Fija la orientación para que los mapas sean reproducibles.
    for j in range(2):
        if componentes[np.argmax(np.abs(componentes[:,j])),j] < 0:
            componentes[:,j] *= -1
    return {'media':mu, 'componentes':componentes, 'ratios':np.maximum(vals[orden],0)/total}

def proyectar(pca,X):
    return np.asarray(X@pca['componentes']) - pca['media']@pca['componentes']

def entrenar(vocab, indices):
    X,V = vocab['X'][:,indices],vocab['V'][:,indices]
    if X.shape[1] == 0:
        raise ValueError('No hay palabras seleccionadas.')
    if (X.shape[0]+V.shape[0])*X.shape[1]*4 > 768*1024**2:
        raise ValueError('La selección requiere demasiada memoria. Aumenta los filtros o excluye más palabras.')
    # API nativa: evita el problema _estimator_type de algunos entornos sklearn/XGBoost.
    nombres = vocab['tabla'].iloc[indices].palabra.tolist()
    dt = xgb.DMatrix(X.toarray(), label=vocab['y'], feature_names=nombres, nthread=2)
    params = dict(objective='binary:logistic',eval_metric='logloss',tree_method='hist',
                  max_depth=4,eta=.1,subsample=1.,colsample_bytree=1.,seed=SEED,nthread=2)
    model = xgb.train(params,dt,num_boost_round=150)
    del dt
    dv = xgb.DMatrix(V.toarray(),feature_names=nombres,nthread=2)
    prob = model.predict(dv)
    y,pred = vocab['yv'],(prob>=.5).astype(int)
    metrics = {'Accuracy':accuracy_score(y,pred),'F1 positivo':f1_score(y,pred,zero_division=0),
               'Precisión positiva':precision_score(y,pred,zero_division=0),
               'Recall positivo':recall_score(y,pred,zero_division=0),'ROC-AUC':roc_auc_score(y,prob)}
    fpr,tpr,_ = roc_curve(y,prob)
    return {'model':model,'metrics':metrics,'cm':confusion_matrix(y,pred,labels=[0,1]),
            'roc':(fpr,tpr),'params':params,'names':nombres,'n_val':len(y)}

def analizar(texto,vocab,indices,resultado):
    limpio = limpiar(texto)
    fila = vocab['vec'].transform([limpio])[:,indices]
    activas = np.flatnonzero(fila.toarray()[0])
    tabla = vocab['tabla'].iloc[indices].iloc[activas].copy()
    prob = float(resultado['model'].predict(xgb.DMatrix(fila.toarray(),feature_names=resultado['names'],nthread=2))[0])
    tokens = set(re.findall(TOKEN,limpio))
    return {'fila':fila,'tabla':tabla,'prob':prob,'tokens':tokens,
            'fuera':sorted(tokens-set(tabla.palabra))}
