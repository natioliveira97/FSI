# MNIST Database

A base de dados MNIST é formada por imagens de digitos escritos a mão. Essa base possui 60000 imagens de treinamento e 10000 imagens de teste. Cada imagem possui 28x28 pixels.

Nesse trabalho de Fundamentos de Sistemas Inteligentes abordamos como funciona três algoritmos de classificação multiclasse: Um-contra-todos (One-vs-Rest), Um-contra-um (One-vc-One) e o código de correção de erros de saída (Error Output Code). Para isso foram treinados cada classificador usando as imagens de treinamento da base MNIST

## Como executar o código

Esse código foi feito usando python v3.5 e precisa de três argumentos de entrada.

* 1: O primeiro argumento define se deseja-se realizar o treinamento ou o test na base de dados, lembrando que para fazer o teste é necessário ter realizado anteriormente o treinamento (--train --test)

* 2: O segundo arguemnto define qual classificador se deseja usar: One-vs-One, One-vs-Rest ou Erro Output Code (--ovo --ovr --eoc)

* 3: O terceiro argumento define qual classificador binário se deseja usar: SVC ou Multilayer Perceptron (--svc --per)

Exemplo de comando:

```
python mnist_base.py --train --ovo --per
```