# PDS – Estudo Dirigido, Parte 2

Fourier, Laplace, Transformada-Z e Análise de Sistemas.

IFPB Campus Campina Grande · Processamento Digital de Sinais · Prof. Moacy Pereira da Silva · 2026.2

Equipe: Aicha Nacambo e Katia Emmanuelle Gomes da Silva

## Estrutura

```
PDS-Parte-2/
|-- relatorio/                 relatório final (Word e PDF)
|-- simulacoes/
|   |-- fourier/               Tópicos 1, 2 e 6
|   |-- dft-fft/               Tópicos 3, 4 e 5
|   |-- nyquist-aliasing/      Tópicos 7 e 8
|   |-- janelamento/           Tópico 9
|   |-- laplace/               Tópicos 10 a 12
|   `-- transformada-z/        Tópicos 13 a 18
|-- mini-projeto/
|   |-- codigo/                scripts do mini projeto
|   |-- dados/                 espectro calculado (CSV)
|   `-- figuras/               figuras do mini projeto
|-- resultados/                saídas numéricas das simulações
`-- README.md
```

## Arquivos

| Arquivo | Conteúdo | Figuras no relatório |
| --- | --- | --- |
| `simulacoes/fourier/topico01_exponenciais.py` | Senoides e exponenciais complexas | 1 e 2 |
| `simulacoes/fourier/topico02_dtft.py` | DTFT de um pulso retangular | 3 |
| `simulacoes/dft-fft/topico03_dft.py` | DFT de duas senoides | 4 |
| `simulacoes/dft-fft/topico04_fft.py` | DFT direta e FFT | 5 |
| `simulacoes/dft-fft/topico05_resolucao.py` | Resolução espectral | 6 |
| `simulacoes/fourier/topico06_freq_normalizada.py` | Frequência normalizada | 7 |
| `simulacoes/nyquist-aliasing/topico07_nyquist.py` | Teorema da amostragem | 8 |
| `simulacoes/nyquist-aliasing/topico08_aliasing.py` | Aliasing | 9 |
| `simulacoes/janelamento/topico09_janelamento.py` | Janelamento e vazamento espectral | 10 |
| `simulacoes/laplace/topico10_laplace.py` | Transformada de Laplace | seção 3.1 |
| `simulacoes/laplace/topico11_funcao_transferencia.py` | Função de transferência (1ª e 2ª ordem) | seção 3.2 |
| `simulacoes/laplace/topico12_polos_zeros_plano_s.py` | Polos, zeros e estabilidade no plano-s | seção 3.3 |
| `simulacoes/transformada-z/topico13_transformada_z.py` | Transformada-Z | seção 3.4 |
| `simulacoes/transformada-z/topico14_regiao_convergencia.py` | Região de convergência | seção 3.5 |
| `simulacoes/transformada-z/topico15_ft_discreta.py` | Função de transferência discreta | seção 3.6 |
| `simulacoes/transformada-z/topico16_estabilidade_plano_z.py` | Estabilidade no plano-z | seção 3.7 |
| `simulacoes/transformada-z/topico17_z_e_dtft.py` | Relação entre Transformada-Z e DTFT | seção 3.8 |
| `simulacoes/transformada-z/topico18_laplace_e_z.py` | Relação entre Laplace e Transformada-Z | seção 3.9 |
| `mini-projeto/codigo/mini_projeto_parte2.py` | Mini projeto, Etapas 2 a 6 (análise espectral) | 11 a 14 |
| `mini-projeto/codigo/mini_projeto_parte2_sistema.py` | Mini projeto, Etapas 7 a 9 (planos s e z) | 15 a 17 |

## Requisitos

Python 3.9 ou superior, com NumPy, Matplotlib e SciPy (já instalados no Google Colab).

```
pip install numpy matplotlib scipy
```

## Como executar

**No Google Colab:** abrir um notebook, copiar o conteúdo de um arquivo `.py` para uma célula e executá-la.

**No computador:** executar cada script a partir da pasta onde ele está.

```
cd simulacoes/fourier
python topico01_exponenciais.py
```

Os scripts do mini projeto são executados a partir da pasta `mini-projeto`:

```
cd mini-projeto
python codigo/mini_projeto_parte2.py
python codigo/mini_projeto_parte2_sistema.py
```

## Saídas

- Os scripts dos Tópicos 1 a 9 e do mini projeto salvam as figuras em PNG numa pasta `figuras/`. Os dos Tópicos 10 a 18 mostram as figuras na tela e alguns também as salvam na pasta onde são executados.
- Todos imprimem os valores numéricos usados no relatório.
- O mini projeto salva também `dados/espectro_fs1000.csv`.
- As saídas numéricas estão registradas em `resultados/`.
- Os tempos do Tópico 4 dependem do computador; os demais valores são sempre os mesmos.
