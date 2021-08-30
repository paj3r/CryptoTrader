# Strateško in taktično upravljanje trgovalne strategije s kriptovalutami
Ta repozitorij vsebuje program za trgovanje s kriptovalutami, narejen za potrebe diplomskega dela na fakulteti za Računalništvo in Informatiko, Univerze v Ljubljani.
# Vsebina repozitorija:
- Strateški del strategije :
  - optimizacija portfoleja na podlagi razmerij Sharpe in Sortino, 
- Taktični del strategije :
  - model ARIMA + GARCH
  - indikator AMA'
  - indikator RSI
- Testiranje s predhodnimi podatki
- Pridobivanje živih podatkov o kriptovalutah
# Zagon programa
Za zagon programa potrebujemo Python 3.9. Ob zagonu programa main.py se začne izvajani test na predhodnih podatkih, ki so locirani v datotekah **1daydata.csv**, ter **4hdata.csv**.
Rezultati testiranja se bodo po končanem testiranju zapisali v mapo **rezultati**.

Če želimo aktualne podatke,lahko pokličemo funkciji *get_1day_data()*, ter *get_4h_data()*, da dobimo aktualne podatke in jih vstavimo v funkciji *strategy()* in *tactics()*.
