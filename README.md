# DMA – TP Caras + ISOMAP

## Setup

```bash
pip install -r requirements.txt
```

## Uso

1. Descargar el dataset desde SharePoint y descomprimirlo dentro de `caras/original/` (las fotos van sueltas, sin subcarpetas):

   <https://alumniiaeedu-my.sharepoint.com/shared?listurl=https%3A%2F%2Falumniiaeedu%2Dmy%2Esharepoint%2Ecom%2Fpersonal%2Fmceriotti%5Fmail%5Faustral%5Fedu%5Far%2FDocuments&id=%2Fpersonal%2Fmceriotti%5Fmail%5Faustral%5Fedu%5Far%2FDocuments%2FCaras&viewid=71522521%2Dbbec%2D4825%2D9802%2D8b3b07729c50>

2. (Opcional) Para test manual, dropear fotos crudas en `caras/original_custom_test/<persona>/`. Para personas no conocidas, usar `caras/original_custom_test/otros/`.

3. Ejecutar el pipeline:

   ```bash
   cd caras
   python procesar.py
   ```

4. Entrenar y evaluar la red:

   ```bash
   python examen.py
   ```

## Resultado

La salida del script muestra % de train, test y comparacion con original_custom_test.
Posibles resultados:
   OK: real y pred fueron acertados.
   DESCARTADO: el pred no paso el umbral, por lo tanto decidimos descartar.
   FAIL: real y pred distintos pero por arriba del umbral. Error del modelo.

   ```bash
========= TEST con umbral de confianza >= 0.9 =========
fila    real            pred            confianza       resultado
0       agustina        belen           0.2588  DESCARTADO
1       agustina        agustina        0.9489  ok
2       agustina        agustina        0.6524  DESCARTADO
3       agustina        agustina        0.9951  ok
4       agustina        agustina        0.9866  ok
5       agustina        fede            0.9587  FAIL
6       agustina        agustina        0.9946  ok
7       belen           belen           0.9825  ok
8       belen           millie          0.2058  DESCARTADO
9       belen           lucia           0.8664  DESCARTADO
10      belen           belen           0.9584  ok
11      belen           millie          0.9097  FAIL
12      belen           belen           0.9944  ok
13      fede            lucia           0.6821  DESCARTADO
14      fede            guillermo       0.1618  DESCARTADO
15      fede            fede            0.9019  ok
16      fede            fede            0.9830  ok
17      fede            guillermo       0.9654  FAIL
18      guillermo       guillermo       0.9993  ok
19      guillermo       guillermo       0.9987  ok
20      guillermo       guillermo       0.9985  ok
21      guillermo       guillermo       0.3792  DESCARTADO
22      guillermo       guillermo       0.9833  ok
23      guillermo       guillermo       0.9982  ok
24      guillermo       guillermo       0.9987  ok
25      guillermo       guillermo       0.9954  ok
26      guillermo       fede            0.1502  DESCARTADO
27      guillermo       guillermo       0.9899  ok
28      guillermo       guillermo       0.9957  ok
29      ignacio         tomas           0.2021  DESCARTADO
30      ignacio         ignacio         0.1562  DESCARTADO
31      ignacio         ignacio         0.9283  ok
32      ignacio         agustina        0.4794  DESCARTADO
33      juani           juani           0.9967  ok
34      juani           juani           0.8582  DESCARTADO
35      juani           juani           0.7514  DESCARTADO
36      juani           juani           0.9981  ok
37      juani           judi            0.1280  DESCARTADO
38      juani           juani           0.9977  ok
39      juani           juani           0.7869  DESCARTADO
40      judi            judi            0.9973  ok
41      judi            juani           0.5622  DESCARTADO
42      judi            ignacio         0.3803  DESCARTADO
43      judi            judi            0.0843  DESCARTADO
44      judi            judi            0.9993  ok
45      judi            judi            0.9905  ok
46      judi            judi            0.9977  ok
47      judi            judi            0.9627  ok
48      judi            judi            0.6539  DESCARTADO
49      judi            judi            0.9968  ok
50      judi            judi            0.9915  ok
51      judi            judi            0.9928  ok
52      judi            judi            0.9998  ok
53      judi            guillermo       0.9865  FAIL
54      lucia           lucia           0.7601  DESCARTADO
55      lucia           juani           0.7400  DESCARTADO
56      lucia           lucia           0.8389  DESCARTADO
57      lucia           lucia           0.9237  ok
58      lucia           lucia           0.0631  DESCARTADO
59      lucia           lucia           0.9984  ok
60      lucia           guillermo       0.8352  DESCARTADO
61      mariangeles     mariangeles     0.9741  ok
62      mariangeles     mariangeles     0.9903  ok
63      mariangeles     mariangeles     0.9116  ok
64      mariangeles     mariangeles     0.9986  ok
65      mariangeles     mariangeles     0.9809  ok
66      mariangeles     martin          0.8606  DESCARTADO
67      martin          martin          0.9985  ok
68      martin          martin          0.9988  ok
69      martin          martin          0.9305  ok
70      martin          martin          0.5623  DESCARTADO
71      martin          martin          0.9985  ok
72      martin          martin          0.9955  ok
73      martin          martin          0.9052  ok
74      martin          martin          0.9985  ok
75      martin          martin          0.9537  ok
76      martin          martin          0.9358  ok
77      martin          martin          0.1245  DESCARTADO
78      martin          martin          0.7992  DESCARTADO
79      matias          martin          0.7794  DESCARTADO
80      matias          matias          0.9941  ok
81      matias          matias          0.8808  DESCARTADO
82      matias          matias          0.9993  ok
83      matias          matias          0.9989  ok
84      matias          matias          0.9975  ok
85      matias          matias          0.9876  ok
86      miguel          miguel          0.9982  ok
87      miguel          miguel          0.9818  ok
88      miguel          miguel          0.9966  ok
89      miguel          miguel          0.9956  ok
90      miguel          miguel          0.9962  ok
91      miguel          miguel          0.9947  ok
92      miguel          miguel          0.9722  ok
93      miguel          fede            0.4781  DESCARTADO
94      miguel          miguel          0.9981  ok
95      miguel          miguel          0.9958  ok
96      miguel          miguel          0.9975  ok
97      millie          millie          0.9852  ok
98      millie          millie          0.9777  ok
99      millie          millie          0.7739  DESCARTADO
100     millie          millie          0.9995  ok
101     millie          millie          0.9986  ok
102     millie          belen           0.9150  FAIL
103     millie          millie          0.9992  ok
104     millie          millie          0.9815  ok
105     millie          millie          0.9953  ok
106     millie          millie          0.9820  ok
107     millie          millie          0.3124  DESCARTADO
108     millie          millie          0.9922  ok
109     millie          millie          0.9630  ok
110     millie          millie          0.6653  DESCARTADO
111     tomas           tomas           0.9982  ok
112     tomas           tomas           0.9918  ok
113     tomas           tomas           0.9960  ok
114     tomas           tomas           0.9064  ok
115     tomas           guillermo       0.1060  DESCARTADO
116     tomas           miguel          0.8922  DESCARTADO

--- Resumen test con umbral ---
Total registros:  117
Descartados:      36/117 (30.77%)
Aceptados:        81/117 (69.23%)
  Aciertos:       76/81 (93.83%)
  Errores:        5/81 (6.17%)

========= TEST CUSTOM con umbral de confianza >= 0.9 =========
fila    real            pred            confianza       resultado
0       matias          matias          0.9894  ok

--- Resumen test custom con umbral ---
Total registros:  1
Descartados:      0/1 (0.00%)
Aceptados:        1/1 (100.00%)
  Aciertos:       1/1 (100.00%)
  Errores:        0/1 (0.00%)
   ```


## Estructura

```
caras/
├── procesar.py
├── examen.py
├── config.py
├── steps/
│   ├── recortar_y_partir_train_test.py
│   ├── recortar_original_custom_test.py
│   ├── entrenar_isomap_y_exportar_txt.py
│   └── reconstruir_caras_isomap.py
├── utils/
│   ├── extraer_nombre_persona.py
│   ├── convertir_heic_a_jpeg.py
│   ├── listar_imagenes_originales.py
│   └── recorte_rostro_dos_ojos.py
├── original/                       # fotos crudas (descargadas de SharePoint)
├── original_custom_test/           # fotos crudas para test manual
│   ├── <persona>/
│   └── otros/
├── caras_1200/                     # generado: recortes 1200x1200
│   ├── training/<persona>/
│   ├── test/<persona>/
│   └── custom_test/<persona>/
└── isomap/                         # generado: embeddings 70D y reconstrucciones
    ├── training_embeddings.txt
    ├── test_embeddings.txt
    ├── test_custom_embeddings.txt
    ├── training/<persona>/
    ├── test/<persona>/
    └── custom_test/<persona>/
```
