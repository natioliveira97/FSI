# Fruits 360

A base fruits 360 é formada por imagens de frutas e vegetais, contendo 120 classes diferentes.

![alt text](https://predictiveprogrammer.com/wp-content/uploads/2018/06/dataset.png)

Para cada classe existem por volta de 500 imagens de treinamento. Cada imagem tem dimensão 100x100 pixels e é colorida. Nas imagens as frutas estão posicionadas de forma diferente e em diversas rotações.

Essa base pode ser baixada pelo link: [https://www.kaggle.com/moltean/fruits](https://www.kaggle.com/moltean/fruits)

Para fins de teste, alteramos a quantidade de classes para treinamento usando os arquivos "data_conf_X.py", mude o import do arquivo em fruits_base.py para alterar o número de classes.

## Como executar

Esse código foi feito usando python v3.5 e precisa de três argumentos de entrada.

* 1: O primeiro argumento define se deseja-se realizar o treinamento ou o test na base de dados, lembrando que para fazer o teste é necessário ter realizado anteriormente o treinamento (--train --test)

* 2: O segundo arguemnto define qual classificador se deseja usar: One-vs-One, One-vs-Rest ou Erro Output Code (--ovo --ovr --eoc)

* 3: O terceiro argumento define qual classificador binário se deseja usar: SVC ou Multilayer Perceptron (--svc --mlp)

Exemplo de comando:

```
python mnist_base.py --train --ovo --svc
```

