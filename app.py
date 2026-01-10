from nicegui import ui
import pandas as pd
import pickle
import numpy as np
import os
import joblib
import copy
import random
import time
from azure.storage.blob import BlobServiceClient





ui.run(
    host='0.0.0.0',
    port=8000,
)
start_time = None      # pas encore démarré
timer_started = False
time_label = None
player_name = ''
name_input = None
name_dialog = None

def load_model_from_blob():
    base_dir = "/home"
    local_model_path = os.path.join(base_dir, MODEL_BLOB_NAME)

    # si déjà téléchargé, on réutilise
    if os.path.exists(local_model_path):
        return joblib.load(local_model_path)

    connect_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    blob_service = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service.get_blob_client(
        container=MODEL_CONTAINER,
        blob=MODEL_BLOB_NAME
    )

    with open(local_model_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    return joblib.load(local_model_path)

def get_model_black():
    global _model_black
    if _model_black is None:
        print("⏳ Chargement du modèle depuis Blob Storage...")
        _model_black = load_model_from_blob()
        print("✅ Modèle prêt")
    return _model_black
modelblack = get_model_black()
LIST_FILE = "listes.pkl"

if os.path.exists(LIST_FILE):
    dflist = joblib.load(LIST_FILE)
else:
    dflist = pd.DataFrame()
TAILLE = 10
tokens = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
newtokens = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
cells = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
state = {
    'rows': None,
    'cols': None,
    'caseactive': False,
    'type' : None,
    'eat' : None,
    'eatsaut' : None,
    'rowsaut' : None,
    'colsaut' : None,
    'transformation' : 0,
    'coup' : 0,
    'ndf' : pd.Series(dtype=object),
    'possauteurrow' : [],
    'possauteurcol' : [],
    'nombresaut': 0,
    'gameover': ' ',
    'nbpion' : 0,
    'nbpionsadverse' : 0,
    'timefinal' : None,
    'tabeatingsaut' : [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)],
    'poslegal' : [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)],
    'deslegal' : [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)],
    'tabeating': [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
}

state['ndf']['typedeplacement'] = ' '
ndfdataframe = pd.DataFrame()

def update_best_results():
    dflist = joblib.load("listes.pkl")

    print(dflist.head())
    print(dflist.shape)

    results_column.clear()

    top10 = get_top_10(dflist)  # 👈 ON PASSE LE BON DF

    if top10.empty:
        ui.label('Aucun résultat').move(results_column)
        return

    for i, row in top10.iterrows():
        ui.label(
            f"{i+1}. {row['joueur']} — {row['time']}"
        ).style(
            'font-size: 16px;'
        ).move(results_column)
        
def get_top_10(dflist):
    if dflist.empty:
        return dflist

    return (
        dflist
        .sort_values(
            by='time',
            key=lambda col: col.map(time_to_seconds),
            ascending=True
        )
        .head(10)
        .reset_index(drop=True)
    )

def time_to_seconds(t):
    m, s = t.split(':')
    return int(m) * 60 + int(s)
    
def reset_timer():
    global start_time, timer_started
    start_time = None
    timer_started = False
    time_label.text = 'Temps : --:--'


def get_elapsed_time():
    elapsed = int(time.time() - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60
    return f'{minutes:02d}:{seconds:02d}'
    
def show_message(text):
    message_label.text = text
    dialog.open()

def update_time():
    if not timer_started:
        return

    elapsed = int(time.time() - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60
    time_label.text = f'Temps : {minutes:02d}:{seconds:02d}'



def init_plateau():
    for row in range(10):
        for col in range(10):

            if (row + col) % 2 != 1:
                continue

            # nettoyage visuel
            cells[row][col].clear()
            tokens[row][col] = None

            # noirs
            if row < 4:
                token = ui.element('div').style(
                    '''
                    width: var(--token-size);
                    height: var(--token-size);
                    border-radius: 50%;
                    background-color: #8B4513;
                    '''
                ).classes('token black')
                token.move(cells[row][col])
                tokens[row][col] = token

            # blancs
            elif row > 5:
                token = ui.element('div').style(
                    '''
                    width: var(--token-size);
                    height: var(--token-size);
                    border-radius: 50%;
                    background-color: #eee;
                    border: 1px solid black;
                    '''
                ).classes('token white')
                token.move(cells[row][col])
                tokens[row][col] = token



def extract_patch(board, row, col, k=1):
    """
    board : np.array (H, W) ex: (10,10)
    row, col : position centrale
    k=1 => patch 3x3
    """
    board = np.array(board, dtype=np.float32)

    # padding pour gérer les bords
    padded = np.pad(
        board,
        pad_width=((k, k), (k, k)),
        mode='constant',
        constant_values=9
    )

    # coordonnées dans le plateau paddé
    r, c = row + k, col + k

    patch = padded[r-k:r+k+1, c-k:c+k+1]
    return patch
def reset_game():
    reset_state()
    init_plateau()
    reset_timer()
    
def reset_state():
    global tokens, state
    TAILLE = 10
    tokens = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
    tokens = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
    cells = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
    state = {
    'rows': None,
    'cols': None,
    'caseactive': False,
    'type' : None,
    'eat' : None,
    'eatsaut' : None,
    'rowsaut' : None,
    'colsaut' : None,
    'transformation' : 0,
    'coup' : 0,
    'ndf' : pd.Series(dtype=object),
    'possauteurrow' : [],
    'possauteurcol' : [],
    'nombresaut': 0,
    'nbpion' : 0,
    'nbpionsadverse' : 0,
    'gameover': ' ',
    'tabeatingsaut' : [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)],
    'poslegal' : [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)],
    'deslegal' : [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)],
    'tabeating': [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
}


    state['ndf']['typedeplacement'] = ' '
 
    
def dame_captures(src_row, src_col, couleur='black'):
    ennemie = 'white' if couleur == 'black' else 'black'
    captures = []

    directions = [(-1,-1), (-1,1), (1,-1), (1,1)]

    for dr, dc in directions:
        r = src_row + dr
        c = src_col + dc
        found_enemy = False
        enemy_pos = None

        while 0 <= r < TAILLE and 0 <= c < TAILLE:
            token = tokens[r][c]

            if token is None:
                if found_enemy:
                    # case valide après capture
                    captures.append((r, c, enemy_pos))
                r += dr
                c += dc
                continue

            # rencontre une pièce
            if ennemie in token._classes and not found_enemy:
                found_enemy = True
                enemy_pos = (r, c)
                r += dr
                c += dc
                continue

            # bloqué (2 ennemis ou ami)
            break

    return captures
def afficher_tabeating(state):
    print('--- CONTENU DE TABEATING ---')
    for r, ligne in enumerate(state['tabeating']):
        for c, cell in enumerate(ligne):
            if cell == ' ' or cell == '':
                print(f'({r},{c}) : vide')
            else:
                print(f'({r},{c}) : {cell}')


def data(tab):
    data = []

    for row in tab:
        data_row = []
        for token in row:
            if token is None:
                data_row.append(0)
            else:
                classes = token._classes

                if 'white' in classes and 'dame' not in classes:
                    data_row.append(1)
                elif 'white' in classes and 'dame' in classes:
                    data_row.append(2)
                elif 'black' in classes and 'dame' not in classes:
                    data_row.append(3)
                elif 'black' in classes and 'dame' in classes:
                    data_row.append(4)
                else:
                    data_row.append(0)  # sécurité

        data.append(data_row)

    return data

def comptegame(tokens, joueur):
  blancs = 0
  noirs = 0

  for ligne in tokens:
        for token in ligne:
            if token is None:
                continue
            if 'white' in token._classes:
                blancs += 1
            elif 'black' in token._classes:
                noirs += 1
  if joueur == 'white':
      pionjoueur = blancs
      pionadverse = noirs
  else:
      pionjoueur = noirs
      pionadverse = blancs
  return pionjoueur, pionadverse
    
def fin_de_partie(tokens, ndfdataframe, joueur):
    final_time = get_elapsed_time()  
    ndfdataframe = ndfdataframe.copy()
    pions, pionsadverse = comptegame(tokens, joueur)
    if joueur == 'white':
        blancs = pions
        noirs = pionsadverse
    else:
        noirs = pions
        blancs = pionsadverse
        
    if blancs == 0 or state['gameover'] == 'white':
      
        ndfdataframe['vainqueur'] = ndfdataframe['joueur'].map({
    'black': 1,
    'white': 0
})
            
        if os.path.exists("dataia3.pkl"):
            df = pd.read_pickle("dataia3.pkl")
            max_partie = df['partie'].max()
            ndfdataframe['partie'] = max_partie + 1
            df = pd.concat(
            [df, ndfdataframe],
             ignore_index=True
             )
            df.to_pickle("dataia3.pkl")
            reset_game()
            return 'noir', final_time

        else:
            ndfdataframe['partie'] = 1
            ndfdataframe.to_pickle("dataia3.pkl")
            reset_game()
            return 'noir', final_time
    if noirs == 0 or state['gameover'] == 'black':
        ndfdataframe['vainqueur'] = ndfdataframe['joueur'].map({
    'black': 1,
    'white': 0
})
        if os.path.exists("dataia3.pkl"):
            df = pd.read_pickle("dataia3.pkl")
            max_partie = df['partie'].max()
            ndfdataframe['partie'] = max_partie + 1
            df = pd.concat(
            [df, ndfdataframe],
             ignore_index=True
             )
            df.to_pickle("dataia3.pkl")
            reset_game()
            return 'blanc', final_time

        else:
            ndfdataframe['partie'] = 1
            ndfdataframe.to_pickle("dataia3.pkl")
            reset_game()
            return 'blanc', final_time
        
    return None, final_time  # la partie continue



def enchainement_dame_noire(token, src_row, src_col):
    state['ndf']['typedeplacement'] = 'mange' 
    cur_row, cur_col = src_row, src_col
    state['ndf']['srcrow'] = src_row
    state['ndf']['srccol'] = src_col
    nombresaut = 0

    while True:
        captures = dame_captures(cur_row, cur_col, 'black')

        if not captures:
            break

        # on prend la première capture possible (IA simple)
        
        dst_row, dst_col, (mid_row, mid_col) = captures[0]
        if nombresaut > 0 :
                   state['ndf']['typedeplacement'] = 'saut' 
                   state['ndf']['nombresaut'] = nombresaut
                   state['possauteurrow'].append(cur_row)
                   state['possauteurcol'].append(cur_col)
                   state['ndf']['rowsaut'] = state['possauteurrow']
                   state['ndf']['colsaut'] = state['possauteurcol']
        tokens[cur_row][cur_col] = None
        cells[mid_row][mid_col].clear()
        tokens[mid_row][mid_col] = None

        token.move(cells[dst_row][dst_col])
        tokens[dst_row][dst_col] = token

        cur_row, cur_col = dst_row, dst_col
        state['ndf']['row'] = cur_row
        state['ndf']['col'] = cur_col
        
        nombresaut += 1


def enchainement_dame(
    token,
    src_row,
    src_col,
    couleur  # 'black' ou 'white'
):
    state['ndf']['typedeplacement'] = 'mange'
    cur_row, cur_col = src_row, src_col
    state['ndf']['srcrow'] = src_row
    state['ndf']['srccol'] = src_col

    nombresaut = 0

    # couleur adverse
    ennemi = 'white' if couleur == 'black' else 'black'

    while True:
        # on récupère toutes les prises possibles
        captures = dame_captures(cur_row, cur_col, couleur)

        if not captures:
            break

        # IA simple : on prend la première capture possible
        dst_row, dst_col, (mid_row, mid_col) = captures[0]

        if nombresaut > 0:
            state['ndf']['typedeplacement'] = 'saut'
            state['ndf']['nombresaut'] = nombresaut
            state['possauteurrow'].append(cur_row)
            state['possauteurcol'].append(cur_col)
            state['ndf']['rowsaut'] = state['possauteurrow']
            state['ndf']['colsaut'] = state['possauteurcol']

        # suppression ancienne position
        tokens[cur_row][cur_col] = None

        # suppression pion mangé
        cells[mid_row][mid_col].clear()
        tokens[mid_row][mid_col] = None

        # déplacement dame
        token.move(cells[dst_row][dst_col])
        tokens[dst_row][dst_col] = token

        # mise à jour position courante
        cur_row, cur_col = dst_row, dst_col
        state['ndf']['row'] = cur_row
        state['ndf']['col'] = cur_col

        nombresaut += 1








def deplacements_possibles(tokens, couleur):
    def est_vide(cell):
        if cell is None:
            return True
        if cell == "vide":
            return True
        # si c'est un objet avec des classes
        classes = getattr(cell, "_classes", None)
        return isinstance(classes, (list, tuple, set)) and ("vide" in classes)

    positions_depart = [[False for _ in range(10)] for _ in range(10)]
    positions_destination = [[False for _ in range(10)] for _ in range(10)]

    # sens des pions
    if couleur == 'black':
        pion_directions = [(1, -1), (1, 1)]
    elif couleur == 'white':
        pion_directions = [(-1, -1), (-1, 1)]
    else:
        raise ValueError("couleur doit être 'black' ou 'white'")

    for row in range(10):
        for col in range(10):
            token = tokens[row][col]
            if est_vide(token):
                continue

            classes = getattr(token, "_classes", [])

            # ne garder que la couleur demandée
            if couleur not in classes:
                continue

            has_move = False

            # ♟️ pion
            if 'pion' in classes or 'token' in classes:  # au cas où ton pion s'appelle "token"
                for dr, dc in pion_directions:
                    new_row = row + dr
                    new_col = col + dc
                    if 0 <= new_row < 10 and 0 <= new_col < 10 and est_vide(tokens[new_row][new_col]):
                        positions_destination[new_row][new_col] = True
                        has_move = True

            # 👑 dame
            elif 'dame' in classes:
                for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    new_row, new_col = row, col
                    while True:
                        new_row += dr
                        new_col += dc
                        if not (0 <= new_row < 10 and 0 <= new_col < 10):
                            break
                        if not est_vide(tokens[new_row][new_col]):
                            break
                        positions_destination[new_row][new_col] = True
                        has_move = True

            if has_move:
                positions_depart[row][col] = True

    return positions_depart, positions_destination
    

def extraire_sauts(tabeatingsaut):
    sauts = []
    for row in range(10):
        for col in range(10):
            cell = tabeatingsaut[row][col]
            if isinstance(cell, str) and 'prise' in cell:
                sauts.append((row, col))
    return sauts
    
def enchainement_pion_noir(token, src_row, src_col, tab):
    cur_row, cur_col = src_row, src_col
    state['ndf']['typedeplacement'] = 'mange'
    state['ndf']['srcrow'] = src_row
    state['ndf']['srccol'] = src_col
    nombresaut = 0
    while True:
        found = False

        for dcol in (-2, 2):
            dst_row = cur_row + 2
            dst_col = cur_col + dcol

            if not (0 <= dst_row < TAILLE and 0 <= dst_col < TAILLE):
                continue

            mid_row = (cur_row + dst_row) // 2
            mid_col = (cur_col + dst_col) // 2

            mid_token = tab[mid_row][mid_col]

            if (
                mid_token
                and 'white' in mid_token._classes
                and tab[dst_row][dst_col] is None
            ):
                # 🔥 PRISE
                if nombresaut > 0 :
                   state['ndf']['typedeplacement'] = 'saut' 
                   state['ndf']['nombresaut'] = nombresaut
                   state['possauteurrow'].append(cur_row)
                   state['possauteurcol'].append(cur_col)
                   state['ndf']['rowsaut'] = state['possauteurrow']
                   state['ndf']['colsaut'] = state['possauteurcol']
                tab[cur_row][cur_col] = None
                cells[mid_row][mid_col].clear()
                tab[mid_row][mid_col] = None

                token.move(cells[dst_row][dst_col])
                tab[dst_row][dst_col] = token

                cur_row, cur_col = dst_row, dst_col
                state['ndf']['row'] = cur_row
                state['ndf']['col'] = cur_col
                
                nombresaut += 1
                found = True
                break  # recommencer depuis la nouvelle position
                

        if not found:
            break
        # 👑 PROMOTION APRÈS LA PRISE
        if cur_row == TAILLE - 1 and 'dame' not in token._classes:
            state['ndf']['transformation'] = 1
            cells[cur_row][cur_col].clear()
            dame = creer_dame_noire()
            dame.move(cells[cur_row][cur_col])
            tab[cur_row][cur_col] = dame
            break  # le pion devient dame → fin de l’enchaînement pion



def enchainement_pion(
    token,
    src_row,
    src_col,
    tab,
    couleur
):
    cur_row, cur_col = src_row, src_col

    # paramètres selon la couleur
    if couleur == 'black':
        dir_row = +2
        ennemi = 'white'
        ligne_promotion = TAILLE - 1
        creer_dame = creer_dame_noire
    else:
        dir_row = -2
        ennemi = 'black'
        ligne_promotion = 0
        creer_dame = creer_dame_blanche

    state['ndf']['typedeplacement'] = 'mange'
    state['ndf']['srcrow'] = src_row
    state['ndf']['srccol'] = src_col

    nombresaut = 0

    while True:
        found = False

        for dcol in (-2, 2):
            dst_row = cur_row + dir_row
            dst_col = cur_col + dcol

            if not (0 <= dst_row < TAILLE and 0 <= dst_col < TAILLE):
                continue

            mid_row = (cur_row + dst_row) // 2
            mid_col = (cur_col + dst_col) // 2

            mid_token = tab[mid_row][mid_col]

            if (
                mid_token
                and ennemi in mid_token._classes
                and tab[dst_row][dst_col] is None
            ):
                # 🔥 PRISE
                if nombresaut > 0:
                    state['ndf']['typedeplacement'] = 'saut'
                    state['ndf']['nombresaut'] = nombresaut
                    state['possauteurrow'].append(cur_row)
                    state['possauteurcol'].append(cur_col)
                    state['ndf']['rowsaut'] = state['possauteurrow']
                    state['ndf']['colsaut'] = state['possauteurcol']

                tab[cur_row][cur_col] = None
                cells[mid_row][mid_col].clear()
                tab[mid_row][mid_col] = None

                token.move(cells[dst_row][dst_col])
                tab[dst_row][dst_col] = token

                cur_row, cur_col = dst_row, dst_col
                state['ndf']['row'] = cur_row
                state['ndf']['col'] = cur_col

                nombresaut += 1
                found = True
                break  # recommencer depuis la nouvelle position

        if not found:
            break

        # 👑 PROMOTION APRÈS LA PRISE
        if cur_row == ligne_promotion and 'dame' not in token._classes:
            state['ndf']['transformation'] = 1
            cells[cur_row][cur_col].clear()
            dame = creer_dame()
            dame.move(cells[cur_row][cur_col])
            tab[cur_row][cur_col] = dame
            break





def afficher_tokens(tokens):
    print('--- CONTENU DE TOKENS ---')
    for r, ligne in enumerate(tokens):
        for c, token in enumerate(ligne):
            if token is None:
                print(f'({r},{c}) : vide')
            else:
                classes = ' '.join(sorted(token._classes))
                print(f'({r},{c}) : {classes}')


def has_capture():
    for ligne in state['tabeating']:
        for cell in ligne:
            if cell.strip():
                return True
    return False

def creer_dame_blanche():
    return ui.element('div').style(
        '''
        width: var(--queen-size);
        height: var(--queen-size);
        border-radius: 60%;
        background-color: #eee;
        '''
    ).classes('dame white')

def creer_dame_noire():
    return ui.element('div').style(
        '''
        width: var(--queen-size);
        height: var(--queen-size);
        border-radius: 60%;
        background-color: #8B4513;
        '''
    ).classes('dame black')
def is_black_cell(row, col):
    return (row + col) % 2 == 1
def lookeating(color, colormange, tab):
    for row, ligne in enumerate(tokens):
      for col, token in enumerate(ligne):
        # on ne garde que les pions blancs
        if token is None or color not in token._classes:
            continue
        if 'dame' not in token._classes:
         if color == 'white':
                 new_row = row - 2
                 if new_row < 0:
                     continue
         if color == 'black':
                 new_row = row + 2
                 if new_row > 9:
                     continue
        #faire les vérif à gauche
         if color == 'white':
             new_col1 = col - 2
         if color == 'black':
             new_col1 = col + 2
         if new_col1 >= 0 and new_col1 <= 9:
            
            # verifier que les cellules apres le pion à manger est vide
            if tokens[new_row][new_col1] is None:
                # regarder sil y a bien un puion noir à manger lors du saut
                
                mid_row = (new_row + row) // 2
                mid_col1 = (new_col1 + col) // 2
                mid_token1 = tokens[mid_row][mid_col1]
                if mid_token1 is not None and colormange in mid_token1._classes:

                    if tab[row][col] == None or tab[row][col] == ' ':
                        tab[row][col] = 'pionselectgauche'
                    else:
                        tab[row][col] =  tab[row][col] + 'pionselectgauche'
                    if tab[new_row][new_col1] == None or tab[new_row][new_col1] == ' ':
                        tab[new_row][new_col1] = 'prise pion gauche'
                    else:
                        tab[new_row][new_col1] = tab[new_row][new_col1] + 'prise pion gauche'
                    
                    

        #faire les vérif à droite
         if color == 'white':
              new_col2 = col + 2
         if color == 'black':
              new_col2 = col - 2    

         if new_col2 <= 9 and new_col2 >= 0:
            
            # verifier que les cellules apres le pion à manger est vide
            if tokens[new_row][new_col2] is None:
                
                # regarder sil y a bien un puion noir à manger lors du saut
                mid_row = (new_row + row) // 2
                mid_col2 = (new_col2 + col) // 2
                mid_token2 = tokens[mid_row][mid_col2]
                if mid_token2 is not None and colormange in mid_token2._classes:
                    if tab[row][col] == None or tab[row][col] == ' ':
                        tab[row][col] = 'pionselectdroite'
                    else:
                        tab[row][col] =  tab[row][col] + 'pionselectdroite'
                    if tab[new_row][new_col2] == None or tab[new_row][new_col2] == ' ':
                        tab[new_row][new_col2] = 'prise pion droite'
                    else:
                        tab[new_row][new_col2] = tab[new_row][new_col2] + 'prise pion droite'
        else:
            #pour les dames y a 4 directions et y a plusieur cellule a regarder
            #verif en haut 
            
            tabrow1 = row - 0
            s = 2
            while s <= tabrow1:
                #verif a gauche
                new_row = row - s
                new_col = col - s
                if new_col >= 0 and new_row >= 0:
                       if tokens[new_row][new_col] is None:
                        #vérifier qu'il n'y a pas de pion dans le saut(exit celui qui est mangé)
                        nombre = s - 2
                        verif = 1
                        while nombre > 0:
                            rowv = row - nombre
                            colv = col - nombre
                            if tokens[rowv][colv] is not None: 
                                verif = 0
                            nombre = nombre - 1
                        mid_row = new_row + 1
                        mid_col = new_col + 1
                        mid_token = tokens[mid_row][mid_col]
                        if mid_token is not None and colormange in mid_token._classes and verif == 1:
                             if tab[row][col] == None or tab[row][col] == ' ':
                                 tab[row][col] = 'dameselecthautgauche'
                             else:
                                 tab[row][col] =  tab[row][col] + 'dameselecthautgauche'
                             if tab[new_row][new_col] == None or tab[new_row][new_col] == ' ':
                                 tab[new_row][new_col] = 'prise dame hautgauche'
                             else:
                                 tab[new_row][new_col] = tab[new_row][new_col] + 'prise dame hautgauche'
                #verif à droite
                new_row = row - s
                new_col = col + s
                if new_col <= 9 and new_row >= 0:
                       if tokens[new_row][new_col] is None:
                        #vérifier qu'il n'y a pas de pion dans le saut(exit celui qui est mangé)
                        nombre = s - 2
                        verif = 1
                        while nombre > 0:
                            rowv = row - nombre
                            colv = col + nombre
                            if tokens[rowv][colv] is not None: 
                                verif = 0 
                            nombre = nombre - 1
                        mid_row = new_row + 1
                        mid_col = new_col - 1
                        mid_token = tokens[mid_row][mid_col]
                        if mid_token is not None and colormange in mid_token._classes and verif == 1:
                             if tab[row][col] == None or tab[row][col] == ' ':
                                 tab[row][col] = 'dameselecthautdroite'
                             else:
                                 tab[row][col] =  tab[row][col] + 'dameselecthautdroite'
                             if tab[new_row][new_col] == None or tab[new_row][new_col] == ' ':
                                 tab[new_row][new_col] = 'prise dame hautdroite'
                             else:
                                 tab[new_row][new_col] = tab[new_row][new_col] + 'prise dame hautdroite'
                s = s + 1
            #verif en bas    
            tabrow2 = 9 - row
            s = 2
            
            while s <= tabrow2:
                #verif a gauche
                new_row = row + s
                new_col = col - s
                if new_col >= 0 and new_row <= 9:
                       if tokens[new_row][new_col] is None:
                        #vérifier qu'il n'y a pas de pion dans le saut(exit celui qui est mangé)
                        nombre = s - 2
                        verif = 1
                        while nombre > 0:
                            rowv = row + nombre
                            colv = col - nombre
                            if tokens[rowv][colv] is not None: 
                                verif = 0
                            nombre = nombre - 1
                        mid_row = new_row - 1
                        mid_col = new_col + 1
                        mid_token = tokens[mid_row][mid_col]
                        if mid_token is not None and colormange in mid_token._classes and verif == 1:
                             if tab[row][col] == None or tab[row][col] == ' ':
                                 tab[row][col] = 'dameselectbasgauche'
                             else:
                                 tab[row][col] =  tab[row][col] + 'dameselectbasgauche'
                             if tab[new_row][new_col] == None or tab[new_row][new_col] == ' ':
                                 tab[new_row][new_col] = 'prise dame basgauche'
                             else:
                                 tab[new_row][new_col] = tab[new_row][new_col] + 'prise dame basgauche'
                #verif à droite
                new_row = row + s
                new_col = col + s
                if new_col <= 9 and new_row <= 9:
                       if tokens[new_row][new_col] is None:
                        #vérifier qu'il n'y a pas de pion dans le saut(exit celui qui est mangé)
                        nombre = s - 2
                        verif = 1
                        while nombre > 0:
                            rowv = row + nombre
                            colv = col + nombre
                            if tokens[rowv][colv] is not None: 
                                verif = 0
                            nombre = nombre - 1
                        mid_row = new_row - 1
                        mid_col = new_col - 1
                        mid_token = tokens[mid_row][mid_col]
                        if mid_token is not None and colormange in mid_token._classes and verif == 1:
                             if tab[row][col] == None or tab[row][col] == ' ':
                                 tab[row][col] = 'dameselectbasdroite'
                             else:
                                 tab[row][col] =  tab[row][col] + 'dameselectbasdroite'
                         
                             if tab[new_row][new_col] == None or tab[new_row][new_col] == ' ':
                                 tab[new_row][new_col] = 'prise dame basdroite'
                             else:
                                 tab[new_row][new_col] =  tab[new_row][new_col] + 'prise dame basdroite'
                             
                s = s + 1


def partiia():
    NB_PARTIES = 60

    for i in range(NB_PARTIES):
      reset_game()
      joueur = 'white'
      adversaire = 'black'

      while 0 == 0:
        state['nombresaut'] = 0
        state['coup'] += 1
        state['ndf']['coup'] = state['coup']
        state['ndf']['dataencour'] = data(tokens)
        pions, pionsadverse = comptegame(tokens, joueur)
        state['ndf']['nbpion'] = pions
        state['ndf']['nbpionsadverse'] = pionsadverse
        state['ndf']['joueur'] = joueur
        positions_depart, positions_destination = deplacements_possibles(tokens, joueur)
        aucun_depart = not any(any(row) for row in positions_depart)
        if aucun_depart and state['eat'] != 1:
              state['gameover'] = joueur
        else:
              move_black_ai_deep_learning(joueur, adversaire)
        state['ndf']['dataaprescoup'] = data(tokens)
        global ndfdataframe
        ndfdataframe = pd.concat(
    [ndfdataframe, state['ndf'].to_frame().T],
    ignore_index=True
)
        state['ndf'] = pd.Series(dtype=object)
        resultat = fin_de_partie(tokens, ndfdataframe, joueur)
        
        if resultat == 'noir':
           ndfdataframe = pd.DataFrame()
           show_message('Victoire des noirs 🖤')
           break
        elif resultat == 'blanc':
           show_message('Victoire des blancs 🤍') 
           ndfdataframe = pd.DataFrame()
           break
        if state['coup'] > 150:
           print('partie trop longue')
           ndfdataframe = pd.DataFrame()
           break
        if joueur == 'black':
           joueur='white'
           adversaire = 'black'
        else:
           joueur = 'black'
           adversaire = 'white'
        


    

    
     





def cell_clic(r, c):
    global start_time, timer_started
    if not timer_started:
        start_time = time.time()
        timer_started = True
    lookeating('white', 'black', state['tabeating'])
    if state['caseactive'] == False:
        selectcase(r, c)
        return
    if state['caseactive'] == True:
        state['rowsaut'] = ' '
        state['colsaut'] = ' '
        if state['eatsaut'] != 1:
            state['coup'] += 1
            state['ndf']['coup'] = state['coup']
            state['ndf']['dataencour'] = data(tokens)
            state['ndf']['joueur'] = 'blanc'
            noirs, blancs = comptegame(tokens, 'white')
            state['ndf']['poslegal'], state['ndf']['deslegal'] = deplacements_possibles(tokens, 'white')
            if uniquement_false(state['ndf']['deslegal']) and state['eat'] != 1:
              state['gameover'] = 'white'
            state['ndf']['nbpion'] = blancs
            state['ndf']['nbpionsadverse'] = noirs
        if state['gameover'] != 'white':
            movejetons(r, c, tokens)
    state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
    lookeating('white', 'black', state['tabeating'])
    state['eatsaut'] = 0
    for row, ligne in enumerate(state['tabeating']):
           for col, cell in enumerate(ligne):
              if cell.strip() != '':
                   if row == state['rowsaut'] and col == state['colsaut'] and 'select' in cell:
                       state['eatsaut'] = 1
    
    if state['eatsaut'] == 1 and state['transformation'] == 0:
            state['nombresaut'] += 1
            state['ndf']['typedeplacement'] = 'saut' 
            state['ndf']['nombresaut'] = state['nombresaut']
            state['possauteurrow'].append(state['rowsaut'])
            state['possauteurcol'].append(state['colsaut'])
            state['ndf']['rowsaut'] = state['possauteurrow']
            state['ndf']['colsaut'] = state['possauteurcol']
    else :
            if state['nombresaut'] == 0: 
                if state['eat'] == 0:
                   state['ndf']['typedeplacement'] = 'deplacement'  
                else:
                   state['ndf']['typedeplacement'] = 'mange'   
            
            state['ndf']['dataaprescoup'] = data(tokens)
            state['ndf']['srcrow'] = state['rows']
            state['ndf']['srccol'] = state['cols']
            state['ndf']['row'] = r
            state['ndf']['col'] = c
            state['ndf']['type'] = state['type']
            state['ndf']['transformation'] = state['transformation']
            global ndfdataframe
            ndfdataframe = pd.concat(
    [ndfdataframe, state['ndf'].to_frame().T],
    ignore_index=True
)           
            state['ndf'] = pd.Series(dtype=object)
            state['possauteurrow'] = []
            state['possauteurcol'] = []
        
    if state['caseactive'] == True and (state['eatsaut'] == 0 or state['transformation'] == 1):
        state['nombresaut'] = 0
        state['coup'] += 1
        state['ndf']['coup'] = state['coup']
        state['ndf']['dataencour'] = data(tokens)
        state['ndf']['joueur'] = 'noir'
        state['ndf']['poslegal'], state['ndf']['deslegal'] = deplacements_possibles(tokens, 'black')
        if uniquement_false(state['ndf']['deslegal']):
             state['gameover'] = 'black'
             
        
        noirs, blancs = comptegame(tokens, 'black')
        if noirs != 0 and state['gameover'] != 'black' and state['gameover'] != 'white':
                 move_black_ai_deep_learning('black', 'white')
                 state['ndf']['dataaprescoup'] = data(tokens)
                 ndfdataframe = pd.concat(
                 [ndfdataframe, state['ndf'].to_frame().T],
                 ignore_index=True
)
        state['ndf'] = pd.Series(dtype=object)
    resultat, timefinal = fin_de_partie(tokens, ndfdataframe, 'black')
    state['timefinal'] = timefinal
    if resultat == 'noir':
       ndfdataframe = pd.DataFrame()
       show_message(f'Victoire des noirs 🖤 en {timefinal}') 
       
    elif resultat == 'blanc':
       ndfdataframe = pd.DataFrame()
       show_message(f'Victoire des blancs 🤍 en {timefinal}') 
       name_dialog.open()
     # reset état
    state['rows'] = None
    state['cols'] = None
    state['caseactive'] = False
    state['type'] = None
    state['eat'] = None
    state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]

   
def uniquement_false(deslegal):
    return all(
        not cell
        for row in deslegal
        for cell in row
    )        
def selectcase(row, col):
      token = tokens[row][col]
      tabmang = 0
      state['eat'] = 0
     
      for r, ligne in enumerate(state['tabeating']):
           for c, cell in enumerate(ligne):
              if cell.strip() != '':
                   tabmang = 1
                   if row == r and col == c and 'select' in cell:
                       state['eat'] = 1

      if state['eatsaut'] == 1 :
          if row != state['rowsaut'] or col != state['colsaut']:
              show_message(f'Vous devez continuer de manger avec row = {state['rowsaut']} col =  {state['colsaut']}')
              state['rows'] = None
              state['cols'] = None
              state['caseactive'] = False
              state['type'] = None
              state['eat'] = None
              state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
              return    

      if tabmang == 1 and state['eat'] == 0:
         state['rows'] = None
         state['cols'] = None
         state['caseactive'] = False
         state['type'] = None
         state['eat'] = None
         state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
         show_message('Vous devez obligatoirement manger !')
         return          
                   
      if 'black' in token._classes:
        state['rows'] = None
        state['cols'] = None
        state['caseactive'] = False
        state['type'] = None
        state['eat'] = None
        state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
        show_message('Vous etes les blancs !')
        return
      if 'dame' in token._classes:
          state['type'] = 'dame'
      else:
          state['type'] = 'pion'
      state['rows'] = row 
      state['cols'] = col
      state['caseactive'] = True
   
def movejetons(row, col, tab):
    state['transformation'] = 0

    if tab[row][col] is not None:
        state['rows'] = None
        state['cols'] = None
        state['caseactive'] = False
        state['type'] = None
        state['eat'] = None
        state['tabeating']= [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
        show_message('case occupée !')
        return
    src_row = state['rows']
    src_col = state['cols']
    typejeton = state['type']
    token = tab[src_row][src_col]
   
        
    if token is None:
        show_message('La case sélectionnée n existe pas !')
        state['rows'] = None
        state['cols'] = None
        state['caseactive'] = False
        state['type'] = None
        state['eat'] = None
        state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
        return  # impossible de sélectionner une case vide

    if state['eat'] == 1:
        #regarder dans le tableau si la case active est une dame ou un pion et quelle est le type de deplacement
        print('saut')
        if 'pionselect' in state['tabeating'][src_row][src_col]: 
            if 'prise pion' in state['tabeating'][row][col] and (('gauche' in state['tabeating'][src_row][src_col] and 'gauche' in state['tabeating'][row][col]) or ('droite' in state['tabeating'][src_row][src_col] and 'droite' in state['tabeating'][row][col])):
                mid_row = (src_row + row) // 2
                mid_col = (src_col + col) // 2
                mid_token = tokens[mid_row][mid_col]
                if mid_token is not None and 'black' in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None
                   
                # retirer la classe de sélection de l’ancienne position
                token.classes(remove='selected')
                # retirer le jeton de l’ancienne case
                tab[src_row][src_col] = None
                # déplacer visuellement le jeton
                token.move(cells[row][col])
                # enregistrer la nouvelle position
                tab[row][col] = token
                state['rowsaut'] = row
                state['colsaut'] = col
            else:
                show_message('Vous devez obligatoirement manger !')
                state['rows'] = None
                state['cols'] = None
                state['caseactive'] = False
                state['type'] = None
                state['eat'] = None
                state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
                return 
        if 'dameselect' in state['tabeating'][src_row][src_col]: 
            manger = 0
            if 'prise dame' in state['tabeating'][row][col] and ('basgauche' in state['tabeating'][src_row][src_col] and 'basgauche' in state['tabeating'][row][col]):
               mid_row = row - 1
               mid_col = col + 1
               mid_token = tab[mid_row][mid_col] 
               if mid_token is not None and 'black' in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None
                    manger = 1
                    state['rowsaut'] = row
                    state['colsaut'] = col
            
            if 'prise dame' in state['tabeating'][row][col] and ('basdroite' in state['tabeating'][src_row][src_col] and 'basdroite' in state['tabeating'][row][col]):
               mid_row = row - 1
               mid_col = col - 1
               mid_token = tab[mid_row][mid_col] 
               if mid_token is not None and 'black' in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None 
                    manger = 1
                    state['rowsaut'] = row
                    state['colsaut'] = col
               
            if 'prise dame' in state['tabeating'][row][col] and ('hautdroite' in state['tabeating'][src_row][src_col] and 'hautdroite' in state['tabeating'][row][col]):
               mid_row = row + 1
               mid_col = col - 1
               mid_token = tab[mid_row][mid_col] 
               if mid_token is not None and 'black' in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None  
                    manger = 1
                    state['rowsaut'] = row
                    state['colsaut'] = col
                
            if 'prise dame' in state['tabeating'][row][col] and ('hautgauche' in state['tabeating'][src_row][src_col] and 'hautgauche' in state['tabeating'][row][col]):

                mid_row = row + 1
                mid_col = col + 1
                mid_token = tab[mid_row][mid_col]
                if mid_token is not None and 'black' in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None
                    manger = 1
                    state['rowsaut'] = row
                    state['colsaut'] = col
            print(f'manger = {manger}')
            if manger == 0:
                show_message('Vous devez obligatoirement manger !')
                state['rows'] = None
                state['cols'] = None
                state['caseactive'] = False
                state['type'] = None
                state['eat'] = None
                state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
                return 
                
            dameblanche = creer_dame_blanche()
            dameblanche.classes(remove='selected')
            # retirer le jeton de l’ancienne case
            tab[src_row][src_col] = None
            cells[src_row][src_col].clear()
            # déplacer visuellement le jeton
            dameblanche.move(cells[row][col])
            tab[row][col] = dameblanche    
        
        
                  
    if typejeton == 'pion' and state['eat'] == 0:                                                    
      if (row == src_row - 1) and (col == src_col + 1 or col == src_col - 1):
          # retirer la classe de sélection de l’ancienne position
          token.classes(remove='selected')
          # retirer le jeton de l’ancienne case
          tab[src_row][src_col] = None
          # déplacer visuellement le jeton
          token.move(cells[row][col])
          # enregistrer la nouvelle position
          tab[row][col] = token
      else:
         show_message('Déplacement interdit !')
         state['rows'] = None
         state['cols'] = None
         state['caseactive'] = False
         state['type'] = None
         state['eat'] = None
         state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
         return  # impossible de se deplacer comme ça
          
    if typejeton == 'dame' and state['eat'] == 0: 
    
        #regarder si le deplacement est autorisé
        if abs(row - src_row) == abs(col - src_col):
            # retirer la classe de sélection de l’ancienne position
            dameblanche = creer_dame_blanche()
            dameblanche.classes(remove='selected')
            # retirer le jeton de l’ancienne case
            tab[src_row][src_col] = None
            cells[src_row][src_col].clear()
             # déplacer visuellement le jeton
            dameblanche.move(cells[row][col])
            tab[row][col] = dameblanche    
                
                
            
        
    

    # transformation en dame
    
    if row == 0:
       dameblanche = creer_dame_blanche()
       tab[row][col].delete()
       dameblanche.move(cells[row][col])
       tab[row][col] = dameblanche
       state['transformation'] = 1

   
def can_move_black(src_row, src_col, dst_row, dst_col, tab):
    # dans la grille
    if not (0 <= dst_row < TAILLE and 0 <= dst_col < TAILLE):
        return False

    # case noire uniquement
    if not is_black_cell(dst_row, dst_col):
        return False

    # destination libre
    if tab[dst_row][dst_col] is not None:
        return False

    return True






def move_black_ai_deep_learning(couleur, couleurenemy):
    state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
    state['ndf']['transformation'] = 0
    lookeating(couleur, couleurenemy, state['tabeating'])
    if has_capture():
     for src_row in range(TAILLE):
        for src_col in range(TAILLE):
            cell = state['tabeating'][src_row][src_col]
            if not cell.strip():
                continue

            token = tokens[src_row][src_col]
            if token is None or couleur not in token._classes:
                continue

            # exemple : prise vers le bas pion
            if 'pion' in cell:
              state['ndf']['type'] = 'pion'
              enchainement_pion(token, src_row, src_col, tokens, couleur)
              
              return
                

            if 'dame' in token._classes:
                state['ndf']['type'] = 'dame'
                enchainement_dame(token, src_row, src_col, couleur)
                
                return
                                   
    else:
      state['ndf']['typedeplacement'] = 'deplacement'
      newtokens = tokens.copy()   # ← avec ()
      row, col, new_row, new_col = nevaluationdeplacement(newtokens, couleur)
      token = tokens[row][col]
      tokens[row][col] = None
      token.move(cells[new_row][new_col])
      tokens[new_row][new_col] = token
      state['ndf']['row'] = new_row
      state['ndf']['col'] = new_col
      state['ndf']['srcrow'] = row
      state['ndf']['srccol'] = col
      state['ndf']['type'] = 'pion'
                    # 👑 PROMOTION EN DAME NOIRE
      if new_row == TAILLE - 1 and couleur == 'black':
                        # supprimer le pion
           state['ndf']['transformation'] = 1
           cells[new_row][new_col].clear()
            # créer la dame noire
           dame = creer_dame_noire()
           dame.move(cells[new_row][new_col])
            # mettre à jour tokens
           tokens[new_row][new_col] = dame
           return  # ⬅️ UN SEUL COUP

      if new_row == 0 and couleur == 'white':
                        # supprimer le pion
           state['ndf']['transformation'] = 1
           cells[new_row][new_col].clear()
            # créer la dame noire
           dame = creer_dame_blanche()
           dame.move(cells[new_row][new_col])
            # mettre à jour tokens
           tokens[new_row][new_col] = dame
           return  # ⬅️ UN SEUL COUP
                    
           


def evaluationdeplacement(tab):

    meilleur_score = float('inf')
    meilleur_coup = None

    for row in range(TAILLE):
        for col in range(TAILLE):

            token = tab[row][col]
            if token is None or 'black' not in token._classes:
                continue

            # =========================
            # PION NOIR
            # =========================
            if 'dame' not in token._classes:
                for dcol in (-1, 1):
                    new_row = row + 1
                    new_col = col + dcol

                    if not can_move_black(row, col, new_row, new_col, tab):
                        continue

                    # 🔹 COPIE LOGIQUE DU PLATEAU
                    tab_test = copy.deepcopy(tab)

                    # 🔹 SIMULATION LOGIQUE (PAS D'UI)
                    tab_test[row][col] = None
                    tab_test[new_row][new_col] = tab[row][col]

                    # 🔹 ENCODAGE POUR LE MODÈLE
                    plateau_avant = np.array(data(tab), dtype=np.float32)   # dataencour
                    plateau_apres = np.array(data(tab_test), dtype=np.float32)   # dataaprescoup


                        
                    patch = extract_patch(
                        plateau_apres,
                        int(new_row),
                        int(new_col),
                        k=3
                        )  # shape (5,5)

                    patch = np.where(patch == 4, 3, patch)    
                    patch_batch = patch[np.newaxis, ..., np.newaxis]
                    score = model.predict(patch_batch).item()
                    # 🔹 COMPARAISON
                    if score < meilleur_score:
                        meilleur_score = score
                        meilleur_coup = (row, col, new_row, new_col)
                    print(f'patch = {patch}')
                    print(f'score = {score}')
                    print(f'meilleur_score = {meilleur_score}')
                    print(f'row = {row}')
                    print(f'col = {col}')
                    print(f'new_row = {new_row}')
                    print(f'new_col = {new_col}')
            # =========================
            # DAME NOIRE
            # =========================
            if 'dame' in token._classes:
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                print('dameintokenclass')
                for dr, dc in directions:
                    new_row = row + dr
                    new_col = col + dc

                    while 0 <= new_row < TAILLE and 0 <= new_col < TAILLE:

                        if tab[new_row][new_col] is not None:
                            break

                        tab_test = copy.deepcopy(tab)
                        tab_test[row][col] = None
                        tab_test[new_row][new_col] = tab[row][col]
                        
                        plateau_avant = np.array(data(tab), dtype=np.float32)   # dataencour
                        plateau_apres = np.array(data(tab_test), dtype=np.float32)   # dataaprescoup

                        patch = extract_patch(
                        plateau_apres,
                        int(new_row),
                        int(new_col),
                        k=3
                        )  # shape (5,5)

                        patch = np.where(patch == 4, 3, patch)
                        patch_batch = patch[np.newaxis, ..., np.newaxis]
                        score = model.predict(patch_batch).item()

                        
                        if score < meilleur_score:
                            meilleur_score = score
                            meilleur_coup = (row, col, new_row, new_col)
                        print(f'patch = {patch}')
                        print(f'score = {score}')
                        print(f'meilleur_score = {meilleur_score}')
                        print(f'row = {row}')
                        print(f'col = {col}')
                        print(f'new_row = {new_row}')
                        print(f'new_col = {new_col}')
                        new_row += dr
                        new_col += dc

    return meilleur_coup

def nevaluationdeplacement(tab, couleur):
    print("nevaluationdeplacement")

    meilleurs = []  # liste de tuples (score, coup)

    # paramètres couleur
    if couleur == 'black':
        couleur_token = 'black'
        dir_pion = +1
        model = modelblack
    else:
        couleur_token = 'white'
        dir_pion = -1
        model = modelwhite

    print(f'couleur_token {couleur_token}')

    for row in range(TAILLE):
        for col in range(TAILLE):

            token = tab[row][col]
            if token is None or couleur_token not in token._classes:
                continue

            # =========================
            # PION
            # =========================
            if 'dame' not in token._classes:
                for dcol in (-1, 1):
                    new_row = row + dir_pion
                    new_col = col + dcol

                    if not can_move_black(row, col, new_row, new_col, tab):
                        continue

                    tab_test = copy.deepcopy(tab)
                    tab_test[row][col] = None
                    tab_test[new_row][new_col] = tab[row][col]

                    plateau_apres = np.array(data(tab_test), dtype=np.float32)

                    patch = extract_patch(
                        plateau_apres,
                        new_row,
                        new_col,
                        k=3
                    )

                    patch = np.where(patch == 4, 3, patch)
                    patch_batch = patch[np.newaxis, ..., np.newaxis]
                    score = model.predict(patch_batch, verbose=0).item()

                    coup = (row, col, new_row, new_col)
                    meilleurs.append((score, coup))
                    meilleurs.sort(key=lambda x: x[0])
                    meilleurs = meilleurs[:2]

                    print(f'score = {score}')
                    print(f'meilleurs = {meilleurs}')

            # =========================
            # DAME
            # =========================
            else:
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

                for dr, dc in directions:
                    new_row = row + dr
                    new_col = col + dc

                    while 0 <= new_row < TAILLE and 0 <= new_col < TAILLE:

                        if tab[new_row][new_col] is not None:
                            break

                        tab_test = copy.deepcopy(tab)
                        tab_test[row][col] = None
                        tab_test[new_row][new_col] = tab[row][col]

                        plateau_apres = np.array(data(tab_test), dtype=np.float32)

                        patch = extract_patch(
                            plateau_apres,
                            new_row,
                            new_col,
                            k=3
                        )

                        patch = np.where(patch == 4, 3, patch)
                        patch_batch = patch[np.newaxis, ..., np.newaxis]
                        score = model.predict(patch_batch, verbose=0).item()

                        coup = (row, col, new_row, new_col)
                        meilleurs.append((score, coup))
                        meilleurs.sort(key=lambda x: x[0])
                        meilleurs = meilleurs[:2]

                        print(f'score = {score}')
                        print(f'meilleurs = {meilleurs}')

                        new_row += dr
                        new_col += dc

    if not meilleurs:
        return None

    # 🎯 choix aléatoire parmi les deux meilleurs
    return random.choice(meilleurs)[1]

def move_black_ai():
    state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
    state['ndf']['transformation'] = 0
    lookeating('black', 'white', state['tabeating'])
    if has_capture():
     for src_row in range(TAILLE):
        for src_col in range(TAILLE):
            cell = state['tabeating'][src_row][src_col]
            if not cell.strip():
                continue

            token = tokens[src_row][src_col]
            if token is None or 'black' not in token._classes:
                continue

            # exemple : prise vers le bas pion
            if 'pion' in cell:
              state['ndf']['type'] = 'pion'
              enchainement_pion_noir(token, src_row, src_col, tokens)
              
              return
                

            if 'dame' in token._classes:
                state['ndf']['type'] = 'dame'

                enchainement_dame_noire(token, src_row, src_col)
                
                return
                                   
    else:
      state['ndf']['typedeplacement'] = 'deplacement'
      for row in range(TAILLE):
        for col in range(TAILLE):

            token = tokens[row][col]
            # on ne regarde que les jetons noirs
            if token is None or 'black' not in token._classes:
                continue

            # destinations possibles (diagonales vers le bas) pion
            if 'dame' not in token._classes:
              for dcol in (-1, 1):
                new_row = row + 1
                new_col = col + dcol

                if can_move_black(row, col, new_row, new_col, tokens):
                    # déplacement
                    
                    
                    tokens[row][col] = None
                    token.move(cells[new_row][new_col])
                    tokens[new_row][new_col] = token
                    state['ndf']['row'] = new_row
                    state['ndf']['col'] = new_col
                    state['ndf']['srcrow'] = row
                    state['ndf']['srccol'] = col
                    state['ndf']['type'] = 'pion'
                    # 👑 PROMOTION EN DAME NOIRE
                    if new_row == TAILLE - 1 :
                        # supprimer le pion
                        state['ndf']['transformation'] = 1
                        cells[new_row][new_col].clear()

                        # créer la dame noire
                        dame = creer_dame_noire()
                        dame.move(cells[new_row][new_col])

                        # mettre à jour tokens
                        tokens[new_row][new_col] = dame
      
                    return  # ⬅️ UN SEUL COUP
                    
            # deplacement dame si pion ne peut pas bouger
            if 'dame' in token._classes:
                
                directions = [(-1,-1), (-1,1), (1,-1), (1,1)]

                for dr, dc in directions:
                  new_row = row + dr
                  new_col = col + dc

                  while 0 <= new_row < TAILLE and 0 <= new_col < TAILLE:

                   # bloqué par une pièce
                     if tokens[new_row][new_col] is not None:
                            break

                     # déplacement possible
                     tokens[row][col] = None
                     token.move(cells[new_row][new_col])
                     tokens[new_row][new_col] = token
                     state['ndf']['row'] = new_row
                     state['ndf']['col'] = new_col
                     state['ndf']['srcrow'] = row
                     state['ndf']['srccol'] = col
                     state['ndf']['type'] = 'dame'
                
                     return  # ⬅️ UN SEUL COUP

                     new_row += dr
                     new_col += dc

      





ui.add_head_html('''
<style>
body {
    margin: 0;
}
</style>
''')

ui.add_head_html('''
<style>
:root {
    --board-size: min(90vmin, 600px);
    --cell-size: calc(var(--board-size) / 10);
    --token-size: calc(var(--cell-size) * 0.7);
}

.center-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}

/* zone plateau + résultats */
.board-layout {
    display: flex;
    gap: 30px;
    align-items: flex-start;
}

/* 📱 mobile : résultats dessous */
@media (max-width: 900px) {
    .board-layout {
        flex-direction: column;
        align-items: center;
    }
}

/* 💻 desktop : résultats à droite */
@media (min-width: 901px) {
    .board-layout {
        flex-direction: row;
    }
}

.title {
    font-size: clamp(42px, 7vw, 72px);
    font-weight: 900;
    letter-spacing: 1px;
    text-shadow:
        2px 2px 0 #ffffffaa,
        4px 4px 0 #00000022;
}
.board {
    border-radius: 14px;
    box-shadow:
        0 12px 30px #00000033,
        inset 0 0 0 6px #8B4513;
    background: #8B4513;
    padding: 6px;
}
.sidebar {
    width: 240px;
    border: 2px solid black;
    border-radius: 10px;
    padding: 15px;
    background-color: #f8f8f8;
}
</style>
''')

ui.add_head_html('''
<style>
:root {
    --queen-size: calc(var(--token-size) * 1.30);
}
</style>
''')

ui.add_head_html('''
<style>
body {
    background: radial-gradient(circle at top, #f7f4ef, #e9e2d9);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
</style>
''')

ui.add_head_html('''
<style>
.title {
    font-size: clamp(42px, 7vw, 72px);
    font-weight: 800;
    margin-bottom: 10px;
    text-align: center;
}
</style>
''')

with ui.dialog().props('persistent') as name_dialog:
    with ui.card().style(
        '''
        width: 320px;
        text-align: center;
        border-radius: 14px;
        padding: 20px;
        '''
    ):
        ui.label('♟️ Jeu de dames').style(
            'font-size: 22px; font-weight: bold; margin-bottom: 10px;'
        )

        ui.label('Entrez votre nom SVP').style(
            'margin-bottom: 15px; font-size: 16px;'
        )

        name_input = ui.input(
            placeholder='Votre nom'
        ).style('width: 100%; margin-bottom: 20px;')

        def validate_name():
            print('validate')
            global player_name
            if name_input.value:
                player_name = name_input.value
                players = pd.DataFrame(columns=['joueur', 'time'])
                players.loc[len(players)] = {
                 'joueur': player_name,
                 'time': state['timefinal']
                }
                print(f'players')
                print(f'{players}')
                print(f'{player_name}')
                print(f'{state['timefinal']}')
                if os.path.exists("listes.pkl"):
                  dfliste = pd.read_pickle("listes.pkl")
                  dfliste = pd.concat(
                  [dfliste, players],
                  ignore_index=True
                  )
                  print(f'dfliste')
                  print(f'{dfliste}')
                  dfliste.to_pickle("listes.pkl")
                else:
                  players.to_pickle("listes.pkl")
                update_best_results()
                name_dialog.close()

        ui.button(
            'Valider',
            on_click=validate_name
        ).style(
            'background-color: #8B4513; color: white; width: 100%;'
        )
# ===== FENÊTRE MODALE (UNE SEULE FOIS) =====
with ui.dialog() as dialog:
    with ui.card().style(
        '''
        width: 300px;
        text-align: center;
        border-radius: 12px;
        '''
    ):
        ui.label('🎉 Message').style(
            'font-size: 22px; font-weight: bold; margin-bottom: 10px;'
        )
        message_label = ui.label('').style('margin-bottom: 20px;')

        ui.button('OK', on_click=dialog.close).style(
            'background-color: #8B4513; color: white;'
        )
with ui.element('div').style(
    '''
    position: fixed;
    inset: 0;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    overflow-y: auto;
    padding: 16px 0;
    '''
):
    # conteneur vertical (titre + plateau)
    with ui.element('div').classes('center-column').style(
     'min-height: fit-content;'
     ):
      
        # 🎯 TITRE
        ui.label('Jeu de dames').classes('title')
         
        # ⏱️ CASE TEMPS
        with ui.element('div').style(
    '''
    border: 2px solid black;
    padding: 6px 15px;
    margin-bottom: 10px;
    border-radius: 8px;
    background-color: #f5f5f5;
    font-size: 13px;
    '''
):
            time_label = ui.label('Temps : --:--')
        with ui.element('div').classes('board-layout'):
  
        # ♟️ PLATEAU
         with ui.element('div').classes('board').style(
            '''
            display: grid;
            grid-template-columns: repeat(10, var(--cell-size));
            grid-template-rows: repeat(10, var(--cell-size));
            border: 2px solid black;
            '''
            ):
            for row in range(TAILLE):
               for col in range(TAILLE):

                 is_black = (row + col) % 2 == 1

                 cell = ui.element('div').style(
                f'''
                width: var(--cell-size);
                height: var(--cell-size);
                background-color: {"black" if is_black else "white"};
                display: flex;
                align-items: center;
                justify-content: center;
                 '''
                 )
             # 🔥 STOCKER LA CELLULE
                 cells[row][col] = cell

            # rendre les cases noires cliquables
                 if is_black:
                      cell.on('click', lambda r=row, c=col: (cell_clic(r, c)))
                      cell.style('cursor: pointer;')

            # jetons noirs (4 premières lignes)
                 if is_black and row < 4:
                     token = ui.element('div').style(
                    '''
                    width: var(--token-size);
                    height: var(--token-size);
                    border-radius: 50%;
                    background-color: #8B4513;
                    '''
                ).classes('token black')
                     token.move(cell)
                     tokens[row][col] = token
            # jetons blancs (4 dernières lignes)
                 if is_black and row > 5:
                     token = ui.element('div').style(
                    '''
                    width: var(--token-size);
                    height: var(--token-size);
                    border-radius: 50%;
                    background-color: #eee;
                    border: 1px solid black;
                    '''
                ).classes('token white')
                     token.move(cell)
                     tokens[row][col] = token

         with ui.element('div').classes('sidebar'):
            ui.label('🏆 Meilleurs résultats').style(
            'font-size: 18px; font-weight: bold; margin-bottom: 10px; max-width: 90vw;'
            )
            results_column = ui.column().style('gap: 8px;')


update_best_results()
ui.timer(1.0, update_time)

ui.run()
