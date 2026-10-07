from pathlib import Path
import hashlib
import html
import io
import json
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
from procesamiento import cargar,vocabulario,ajustar_pca,proyectar,entrenar,analizar,TOKEN

st.set_page_config(page_title='Reseñas · WoE · XGBoost',page_icon='💬',layout='wide')
ROOT = Path(__file__).resolve().parent
POS,NEG = '#147d92','#cf603b'
COLORS = {'Positiva':POS,'Negativa':NEG}
st.title('Reseñas: de las palabras a la predicción')
st.caption('Selecciona vocabulario, explora su estructura y analiza nuevas reseñas en inglés.')

@st.cache_resource(max_entries=2,show_spinner=False)
def base_cache(path,version):
    return cargar(path)

@st.cache_resource(max_entries=3,show_spinner=False)
def vocab_cache(identity,frecuencia,_base):
    return vocabulario(_base,frecuencia)

@st.cache_resource(max_entries=3,show_spinner=False)
def pca_cache(signature,indices,_vocab):
    pca = ajustar_pca(_vocab['X'][:,indices])
    Z = proyectar(pca,_vocab['X'][:,indices]) if pca is not None else None
    return pca,Z

@st.cache_data(max_entries=4,show_spinner=False)
def nube_cache(words,counts,woes):
    color = {w:POS if v>=0 else NEG for w,v in zip(words,woes)}
    wc = WordCloud(width=1400,height=550,background_color='white',max_words=150,
                   random_state=42,collocations=False).generate_from_frequencies(dict(zip(words,counts)))
    wc.recolor(color_func=lambda word,**kwargs:color[word],random_state=42)
    buff=io.BytesIO();wc.to_image().save(buff,format='PNG');return buff.getvalue()

path = ROOT / 'IMDB Dataset.zip'
if not path.exists():
    st.info('Coloca IMDB Dataset.csv en esta carpeta y vuelve a cargar la página:')
    st.code(str(ROOT));st.stop()
identity=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
try:
    with st.spinner('Leyendo y limpiando reseñas…'):
        base=base_cache(str(path),identity)
except Exception as exc:
    st.error(str(exc));st.stop()
with st.sidebar:
    st.header('1 · Filtros iniciales')
    with st.form('filtros'):
        wmin=st.number_input('|WoE| mínimo',min_value=0.,value=.36,step=.01,format='%.2f')
        fmin=st.number_input('Frecuencia mínima (reseñas)',min_value=1,value=2500,step=100)
        st.form_submit_button('Aplicar filtros',use_container_width=True)
    st.caption('Frecuencia = reseñas de entrenamiento con la palabra; no es la cantidad de palabras del vocabulario.')
    st.caption('WoE y selección: entrenamiento (80%). Indicadores: validación (20%). Semilla: 42.')
    st.info('Los textos deben estar en inglés, como el dataset IMDb.')
try:
    with st.spinner('Calculando vocabulario y WoE de entrenamiento…'):
        vocab=vocab_cache(identity,int(fmin),base)
except Exception as exc:
    st.warning(str(exc));st.stop()
tabla=vocab['tabla']
pasan=tabla.loc[tabla.woe.abs()>=float(wmin)-1e-12].copy()
st.session_state.setdefault('omitidas',[])
st.session_state.setdefault('epoch',0)
# La identidad del modelo depende de filtros, columnas y archivo.
st.subheader('2 · Elegir palabras que entran al análisis')
st.caption('Solo puedes agregar a exclusiones palabras que pasaron los filtros. Las exclusiones se mantienen al cambiar los umbrales hasta que las restaures.')
a,b,c=st.columns(3)
a.metric('Reseñas limpias',f"{len(base['df']):,}")
b.metric('Entrenamiento',f"{len(base['it']):,}")
c.metric('Validación',f"{len(base['iv']):,}")
activas=pasan.loc[~pasan.palabra.isin(st.session_state.omitidas)].copy()
with st.form('excluir'):
    elegidas=st.multiselect('Palabras incluidas que deseas excluir',options=activas.sort_values('frecuencia',ascending=False).palabra.tolist(),key=f"excluir_{st.session_state.epoch}")
    if st.form_submit_button('Excluir seleccionadas') and elegidas:
        st.session_state.omitidas=sorted(set(st.session_state.omitidas)|set(elegidas))
        st.session_state.epoch+=1;st.rerun()
with st.expander(f"Excluidas manualmente: {len(st.session_state.omitidas)} · recuperar o importar"):
    with st.form('recuperar'):
        devolver=st.multiselect('Recuperar palabras',st.session_state.omitidas)
        if st.form_submit_button('Recuperar seleccionadas') and devolver:
            st.session_state.omitidas=sorted(set(st.session_state.omitidas)-set(devolver));st.rerun()
    if st.button('Restaurar todas las exclusiones'):
        st.session_state.omitidas=[];st.rerun()
    upload=st.file_uploader('Importar JSON de exclusiones o vocabulario final',type='json')
    if st.button('Aplicar exclusiones del JSON',disabled=upload is None):
        try:
            obj=json.loads(upload.getvalue());vals=obj if isinstance(obj,list) else obj['omitted_words']
            if not isinstance(vals,list) or not all(isinstance(w,str) for w in vals):raise ValueError('Se requiere una lista de palabras.')
            st.session_state.omitidas=sorted(set(w.strip().lower() for w in vals));st.rerun()
        except Exception as exc:st.error(str(exc))
indices=tuple(activas.index.astype(int).tolist())
signature=hashlib.sha256(repr((identity,fmin,wmin,indices)).encode()).hexdigest()
if st.session_state.get('model_signature')!=signature:
    st.session_state.pop('resultado',None)
    st.session_state.pop('texto_analizado',None)
a,b,c=st.columns(3)
a.metric('Pasaron filtros',len(pasan));b.metric('Excluidas entre las candidatas',len(pasan)-len(activas));c.metric('Palabras finales',len(activas))
config={'filters':{'minimum_abs_woe':float(wmin),'minimum_document_frequency':int(fmin)},
        'included_words':activas.palabra.tolist(),'omitted_words':st.session_state.omitidas,
        'representation':'binary presence 1 / absence 0','statistics_source':'training',
        'woe_convention':'ln(((positive_presence+0.5)/(positive_reviews+1))/((negative_presence+0.5)/(negative_reviews+1)))',
        'token_pattern':TOKEN,'random_seed':42,'validation_fraction':.2,
        'word_statistics':activas.to_dict('records')}
st.download_button('Descargar vocabulario final JSON',json.dumps(config,ensure_ascii=False,indent=2),file_name='vocabulario_final.json',mime='application/json')
with st.expander('Ver únicamente las palabras incluidas'):
    st.dataframe(activas.sort_values('frecuencia',ascending=False),hide_index=True,use_container_width=True)
if not indices:
    st.warning('No quedan palabras. Reduce los filtros o recupera exclusiones.');st.stop()

st.subheader('3 · PCA y nube de palabras')
st.caption('Mismo vocabulario para PCA, nube y XGBoost. PCA centrado, sin estandarizar, ajustado solo con entrenamiento; XGBoost usa las binarias originales.')
pca,Z=None,None
try:
    with st.spinner('Calculando PCA del vocabulario actual…'):
        pca,Z=pca_cache(signature,indices,vocab)
except Exception as exc:st.warning(f'No se pudo calcular PCA: {exc}')

def mapa(nueva=None):
    rng=np.random.default_rng(42)
    positions=np.sort(rng.choice(len(Z),min(3000,len(Z)),replace=False))
    data=pd.DataFrame({'PC1':Z[positions,0],'PC2':Z[positions,1],
        'Sentimiento':np.where(vocab['y'][positions]==1,'Positiva','Negativa'),
        'Reseña':base['df'].review.iloc[base['it'][positions]].astype(str).str.slice(0,200).to_numpy()})
    fig=px.scatter(data,x='PC1',y='PC2',color='Sentimiento',color_discrete_map=COLORS,opacity=.38,hover_data=['Reseña'])
    if nueva is not None:
        punto=proyectar(pca,nueva)[0]
        fig.add_trace(go.Scatter(x=[punto[0]],y=[punto[1]],mode='markers',name='Nueva reseña (predicción)',
            marker=dict(size=20,color='#111827',symbol='star',line=dict(color='white',width=2)),
            hovertemplate='Nueva reseña<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'))
    fig.update_layout(height=470,xaxis_title=f"PC1 ({100*pca['ratios'][0]:.2f}%)",
                      yaxis_title=f"PC2 ({100*pca['ratios'][1]:.2f}%)")
    return fig
l,r=st.columns(2)
with l:
    if pca is None:st.info('El PCA 2D requiere al menos dos palabras y varianza distinta de cero.')
    else:
        st.plotly_chart(mapa(),use_container_width=True,key='mapa_base')
        st.caption(f"PC1 + PC2 explican {100*pca['ratios'].sum():.2f}% de la varianza. Se dibujan hasta 3,000 reseñas de entrenamiento.")
with r:
    png=nube_cache(tuple(activas.palabra),tuple(activas.frecuencia),tuple(activas.woe))
    st.image(png,use_container_width=True)
    st.caption('Hasta 150 palabras. Tamaño = frecuencia. Azul = WoE positivo; naranja = WoE negativo.')
    st.download_button('Descargar nube PNG',png,file_name='nube_palabras.png',mime='image/png')

st.subheader('4 · Entrenar XGBoost')
st.caption('150 árboles · profundidad 4 · learning rate 0.10 · umbral de clasificación 0.50. Cambiar filtros o exclusiones invalida el modelo anterior.')
if st.button('Entrenar / actualizar XGBoost',type='primary'):
    try:
        with st.spinner('Entrenando XGBoost y evaluando en validación…'):
            st.session_state.resultado=entrenar(vocab,list(indices))
            st.session_state.model_signature=signature
    except Exception as exc:st.error(f'No se pudo entrenar: {exc}')
resultado=st.session_state.get('resultado')
if resultado is None:
    st.info('Entrena el modelo con la selección actual para ver indicadores y analizar un texto.');st.stop()
for col,(nombre,valor) in zip(st.columns(5),resultado['metrics'].items()):col.metric(nombre,f'{valor:.3f}')
st.caption(f"Indicadores sobre {resultado['n_val']:,} reseñas de validación. F1, precisión y recall toman positive como clase positiva. Son resultados exploratorios: ajustar repetidamente con esta validación no reemplaza una prueba final independiente.")
l,r=st.columns(2)
with l:
    cm=px.imshow(resultado['cm'],x=['Negativa','Positiva'],y=['Negativa','Positiva'],text_auto=True,
                 labels={'x':'Predicción','y':'Clase real','color':'Reseñas'},color_continuous_scale='Blues',title='Matriz de confusión')
    st.plotly_chart(cm,use_container_width=True)
with r:
    fpr,tpr=resultado['roc'];fig=go.Figure(go.Scatter(x=fpr,y=tpr,mode='lines',name='XGBoost'))
    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash',color='gray'))
    fig.update_layout(title='Curva ROC',xaxis_title='Tasa de falsos positivos',yaxis_title='Tasa de verdaderos positivos')
    st.plotly_chart(fig,use_container_width=True)
st.download_button('Descargar indicadores CSV',pd.DataFrame([resultado['metrics']]).to_csv(index=False),file_name='indicadores.csv',mime='text/csv')

st.subheader('5 · Analizar una nueva reseña')
with st.form('resena'):
    texto=st.text_area('Escribe una reseña en inglés',height=120,placeholder='The acting was excellent, but the story was boring and disappointing.')
    enviado=st.form_submit_button('Analizar reseña',type='primary')
if enviado:
    if texto.strip():st.session_state.texto_analizado=texto
    else:st.session_state.pop('texto_analizado',None);st.warning('Escribe un texto antes de analizar.')
if st.session_state.get('texto_analizado'):
    texto=st.session_state.texto_analizado
    analisis=analizar(texto,vocab,list(indices),resultado)
    detalle=analisis['tabla'].copy();prob=analisis['prob']
    st.write('**Texto analizado:**',texto)
    if detalle.empty:
        st.warning('La reseña no contiene ninguna palabra del vocabulario final. El vector es todo ceros; la predicción carece de evidencia léxica reconocida. Prueba otro texto o revisa los filtros.')
    a,b,c=st.columns(3)
    a.metric('Predicción','Positiva' if prob>=.5 else 'Negativa')
    b.metric('Probabilidad estimada positiva',f'{100*prob:.1f}%')
    c.metric('Palabras del modelo presentes',len(detalle))
    st.caption('Probabilidad estimada por XGBoost; no es una probabilidad calibrada. La reseña nueva no modifica el entrenamiento ni las estadísticas.')
    if pca is not None:
        st.plotly_chart(mapa(analisis['fila']),use_container_width=True,key='mapa_nueva')
        st.caption('★ La nueva reseña se proyecta con el mismo PCA, sin volver a ajustarlo. Su cercanía visual no equivale a la predicción de XGBoost.')
    if not detalle.empty:
        st.markdown('**Palabras reconocidas en la reseña**')
        pesos=detalle.set_index('palabra').woe.to_dict()
        partes=re.split(r'(\b[a-zA-Z]{2,}\b)',texto)
        marcado=''.join(f'<mark style="background:{POS if pesos[p.lower()]>=0 else NEG};color:white;padding:2px 4px;border-radius:4px">{html.escape(p)}</mark>' if p.lower() in pesos else html.escape(p) for p in partes)
        st.markdown('<div style="line-height:2;padding:14px;border:1px solid #ddd;border-radius:8px">'+marcado+'</div>',unsafe_allow_html=True)
        st.caption('Solo se colorean palabras presentes que forman parte del modelo. Las demás no son necesariamente neutrales: están fuera de su vocabulario.')
        orden=detalle.assign(abs_woe=detalle.woe.abs()).sort_values('abs_woe',ascending=False)
        limite=st.slider('Máximo de palabras a mostrar en los gráficos',min_value=5,max_value=100,value=30,step=5)
        vista=orden.head(limite).sort_values('woe')
        st.caption(f'Se muestran {len(vista)} de {len(detalle)} palabras reconocidas, priorizando |WoE|. La tabla y el CSV contienen todas.')
        fig=go.Figure(go.Bar(x=vista.woe,y=vista.palabra,orientation='h',marker_color=[POS if w>=0 else NEG for w in vista.woe],
             hovertemplate='%{y}<br>WoE: %{x:.3f}<extra></extra>'))
        fig.add_vline(x=0,line_color='gray')
        fig.update_layout(title='Asociación de las palabras presentes: WoE de entrenamiento',xaxis_title='WoE (negativo ← 0 → positivo)',
                          height=max(340,len(vista)*27+100))
        st.plotly_chart(fig,use_container_width=True)
        st.info('El WoE describe la asociación histórica de cada palabra. No es su contribución individual a la predicción de XGBoost: el modelo puede combinar palabras y aprender interacciones.')
        barras=go.Figure()
        for label,col,color in [('Negativas','negativas',NEG),('Positivas','positivas',POS)]:
            valores=100*vista[col]/vista.frecuencia
            barras.add_trace(go.Bar(name=label,y=vista.palabra,x=valores,orientation='h',marker_color=color,
                customdata=vista[col],text=[f'{x:.1f}%' for x in valores],textposition='inside',
                hovertemplate='%{y}<br>'+label+': %{x:.1f}%<br>Reseñas: %{customdata:,.0f}<extra></extra>'))
        barras.update_layout(barmode='stack',title='Entre las reseñas que contienen cada palabra, ¿qué sentimiento tienen?',
            xaxis=dict(title='Porcentaje dentro de las reseñas con esa palabra',range=[0,100]),
            height=max(340,len(vista)*30+100),legend=dict(orientation='h'))
        st.plotly_chart(barras,use_container_width=True)
        st.caption('Cada barra suma 100%: positivas / (positivas + negativas) y negativas / (positivas + negativas), usando solo entrenamiento. No son las tasas de presencia por clase usadas para calcular WoE; esas tasas aparecen en la tabla.')
        with st.expander('Tabla completa: frecuencias, WoE y tasas de presencia por clase'):
            st.dataframe(detalle,hide_index=True,use_container_width=True)
        st.download_button('Descargar análisis de palabras CSV',detalle.to_csv(index=False).encode('utf-8-sig'),file_name='palabras_resena.csv',mime='text/csv')
    with st.expander(f"Palabras fuera del vocabulario final ({len(analisis['fuera'])})"):
        st.write(', '.join(analisis['fuera']) or 'Ninguna.')
