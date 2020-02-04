# Car Base

Iremos começar aprendendo os classificadores multiclasse com uma base simples da UCI Machine Learning Repository.

Essa base possui apenas 4 classes e 1728 exemplos de teste.

Os atributos da base são:

* buying: vhigh, high, med, low. (Custo de compra)
* maint: vhigh, high, med, low. (Custo de manutenção)
* doors: 2, 3, 4, 5more. (Quantidade de portas)
* persons: 2, 4, more. (Quantidade de lugares)
* lug_boot: small, med, big. (Tamanho do porta malas)
* safety: low, med, high. (Segurança)

Todos os atributos são categóricos

As classes são:

* unacc (Inaceitável)
* acc (Aceitável)
* good (Bom)
* vgood (Muito bom)

A quantidade de amostras de cada classe varia bastante, a base é desbalanceada.

## Como executar

Resolvemos o problema usando 3 abordagens de problema multiclasse:

* Um-contra-todos (--ovr)
* Um-contra-um (--ovo)
* Código de correção de erros de saída (--eoc)

Como classificador binário, demos as opções de escolher:

* Suport Vector Machine (svc)
* Multilayer Perceptron (--mlp)

Para executar o código rode o script dando como argumentos de entrada o classificador multiclasse e o binário.

```
python car_base.py --ovo --mlp
```