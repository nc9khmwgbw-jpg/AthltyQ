# Position Mapping Report

## Sources analysées

- `DATA_PIPELINE/NETTOYAGE/data/dataset_v2_injury.csv`: aucune colonne position fiable.
- `DATA_PIPELINE/NETTOYAGE/data/merged_dataset_clean.csv`: aucune colonne position.
- `DATA_PIPELINE/SCRAPPING/data/raw/transfermarkt/injury_history.csv`: colonne `Position`, source utilisée.
- `DATA_PIPELINE/SCRAPPING/data/raw/sofascore/**/*.csv`: fichiers match/stats sans colonne position exploitable.

## Stratégie de mapping

1. Conserver `Position` existante si elle est déjà connue dans le dataset d'entrée.
2. Fallback Transfermarkt `injury_history.csv` via nom joueur normalisé.
3. Exclure les noms homonymes avec plusieurs postes Transfermarkt contradictoires.
4. Fallback `Unknown` si aucune source fiable n'est disponible.

## Couverture

- Joueurs totaux: `3,852`
- Joueurs avec position avant mapping: `0`
- Joueurs avec position après mapping: `3,202`
- Joueurs encore Unknown: `650`
- Taux de couverture après mapping: `83.13%`

## Distribution des sources

| Position_Source          |   players |
|:-------------------------|----------:|
| injury_history           |      3202 |
| unknown                  |       645 |
| ambiguous_injury_history |         5 |

## Distribution des postes

| Position   |   players |
|:-----------|----------:|
| CB         |       599 |
| CF         |       492 |
| CM         |       485 |
| DM         |       291 |
| RB         |       272 |
| LW         |       267 |
| RW         |       245 |
| AM         |       233 |
| LB         |       221 |
| RM         |        38 |
| LM         |        22 |
| SS         |        17 |
| GK         |        13 |
| Attack     |         4 |
| Defender   |         3 |

## Exemples de mappings

| Nom                   | Team              | League         | Position   | Position_Source   | Position_Ambiguous   |
|:----------------------|:------------------|:---------------|:-----------|:------------------|:---------------------|
| Aaron Bibout          | KRC Genk          | ProLeague      | CF         | injury_history    | False                |
| Aaron Herrera         | DC United         | MLS            | RB         | injury_history    | False                |
| Aaron Hickey          | Brentford         | Premier        | RB         | injury_history    | False                |
| Aaron Muirhead        | Arbroath          | ScottishPrem   | RB         | injury_history    | False                |
| Aaron Tshibola        | Kilmarnock        | ScottishPrem   | CM         | injury_history    | False                |
| Aarón Martín          | Genoa             | SerieA         | LB         | injury_history    | False                |
| Abdallah Sima         | Rc Lens           | Ligue 1        | LW         | injury_history    | False                |
| Abdel Abqar           | Getafe            | LaLiga         | CB         | injury_history    | False                |
| Abdelkader Bedrane    | Damac FC          | SaudiProLeague | CB         | injury_history    | False                |
| Abdelkahar Kadri      | KAA Gent          | ProLeague      | AM         | injury_history    | False                |
| Abdelmounaim Boutouil | Al-Hazem          | SaudiProLeague | CB         | injury_history    | False                |
| Abderrahman Rebbach   | Deportivo Alaves  | LaLiga         | LW         | injury_history    | False                |
| Abdessamad Ezzalzouli | Real Betis        | LaLiga         | LW         | injury_history    | False                |
| Abdoul Ouattara       | Rc Strasbourg     | Ligue 1        | RB         | injury_history    | False                |
| Abdoulaye Doucouré    | Neom SC           | SaudiProLeague | AM         | injury_history    | False                |
| Abdoulaye Faye        | Lorient           | Ligue 1        | CB         | injury_history    | False                |
| Abdoulaye Sissako     | Sint-Truidense VV | ProLeague      | DM         | injury_history    | False                |
| Abdoulaye Touré       | Le Havre          | Ligue 1        | DM         | injury_history    | False                |
| Abdu Conté            | Casa Pia          | LigaPortugal   | LB         | injury_history    | False                |
| Abdukodir Khusanov    | Manchester City   | Premier        | CB         | injury_history    | False                |
| Abdulay Juma Bah      | Nice              | Ligue 1        | CB         | injury_history    | False                |
| Abdulaziz Al-Aliwa    | Al-Kholood        | SaudiProLeague | LW         | injury_history    | False                |
| Abdulaziz Al-Dwehe    | Al-Hazem          | SaudiProLeague | AM         | injury_history    | False                |
| Abdulaziz Al-Harbi    | Al-Hazem          | SaudiProLeague | CB         | injury_history    | False                |
| Abdulaziz Noor        | Neom SC           | SaudiProLeague | RW         | injury_history    | False                |

## Exemples encore Unknown

| Nom                   | Team           | League          | Position   | Position_Source   | Position_Ambiguous   |
|:----------------------|:---------------|:----------------|:-----------|:------------------|:---------------------|
| Abderazak Hamdallah   | Al-Shabab      | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdul Fatawu Issahaku | Leicester City | Championship    | Unknown    | unknown           | False                |
| Abdulaziz Al Bishi    | Al-Ittihad     | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdulaziz Al Fawaz    | Al-Fateh       | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdulaziz Al Hatila   | Al-Okhdood     | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdulaziz Al-Swealem  | Al-Fateh       | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdulelah Al Khaibari | Al-Riyadh      | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdulelah Alshamary   | Al-Najma SC    | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdulmalik Al Oyayari | Neom SC        | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdulrahman Al-Dosari | Al-Kholood     | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdulrahman Alobud    | Al-Ittihad     | SaudiProLeague  | Unknown    | unknown           | False                |
| Abdurahman Al-Dakheel | Al-Hazem       | SaudiProLeague  | Unknown    | unknown           | False                |
| Adam Markhiev         | 1. FC Nurnberg | Bundesliga2     | Unknown    | unknown           | False                |
| Adama Malouda Traoré  | Genclerbirligi | SuperLigTurquie | Unknown    | unknown           | False                |
| Adil Aouchiche        | FC Schalke 04  | Bundesliga2     | Unknown    | unknown           | False                |
| Adrian Gantenbein     | FC Schalke 04  | Bundesliga2     | Unknown    | unknown           | False                |
| Adriano Grimaldi      | 1. FC Nurnberg | Bundesliga2     | Unknown    | unknown           | False                |
| Ahmad Asiri           | Al-Khaleej     | SaudiProLeague  | Unknown    | unknown           | False                |
| Ahmadou Bamba Dieng   | Lorient        | Ligue 1         | Unknown    | unknown           | False                |
| Ahmed Al Siyahi       | Al-Riyadh      | SaudiProLeague  | Unknown    | unknown           | False                |
| Ahmed Bamasud         | Al-Fayha       | SaudiProLeague  | Unknown    | unknown           | False                |
| Aiden  Hezarkhani     | Real Salt Lake | MLS             | Unknown    | unknown           | False                |
| Ala'a Al-Hejji        | Neom SC        | SaudiProLeague  | Unknown    | unknown           | False                |
| Alan                  | Moreirense     | LigaPortugal    | Unknown    | unknown           | False                |
| Albert Guðmundsson    | Fiorentina     | SerieA          | Unknown    | unknown           | False                |

## Homonymes / positions contradictoires

| position_key   | positions   |   position_count |
|:---------------|:------------|-----------------:|
| andre silva    | CF, DM      |                2 |
| antony         | LW, RW      |                2 |
| guga           | CM, RB      |                2 |
| joao pedro     | CF, CM      |                2 |
| pedro          | LW, RW      |                2 |
