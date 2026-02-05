from nicegui import ui
import pandas as pd
import pickle
import numpy as np
import os
import joblib
import copy
import random
import time
import tensorflow as tf
import keras
import joblib
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Configuration

connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  # Récupère la connection string depuis les variables d'environnement

# Initialisation du client Blob Storage
blob_service = BlobServiceClient.from_connection_string(connect_str)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model_to_blob(model, blob_name, container_name, blob_service):
    """
    Sauvegarde un modèle/DataFrame dans Azure Blob Storage.

    Args:
        model: Objet à sauvegarder (modèle sklearn, DataFrame pandas, etc.)
        blob_name (str): Nom du fichier dans le blob (ex: "listes.pkl")
        container_name (str): Nom du conteneur Azure
        blob_service: Instance de BlobServiceClient

    Returns:
        bool: True si succès, False sinon
    """
    try:
        # 1. Sérialisation en mémoire
        buffer = BytesIO()
        
        # Choix automatique entre pickle pandas ou joblib
        if hasattr(model, 'to_pickle'):  # Si c'est un DataFrame pandas
            model.to_pickle(buffer)
        else:  # Sinon utiliser joblib (pour les modèles sklearn)
            import joblib
            joblib.dump(model, buffer)
        
        buffer.seek(0)

        # 2. Upload vers Azure
        blob_client = blob_service.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        blob_client.upload_blob(buffer, overwrite=True)

        logging.info(f"✅ {blob_name} sauvegardé avec succès dans {container_name}")
        return True

    except Exception as e:
        logging.error(f"❌ Échec de la sauvegarde de {blob_name}: {str(e)}", exc_info=True)
        return False




def load_model_from_blob(blob_name, MODEL_CONTAINER, blob_service, default=None):
    """
    Charge un modèle depuis Azure Blob Storage avec gestion des erreurs.

    Args:
        blob_name (str): Nom du fichier modèle dans le conteneur.
        MODEL_CONTAINER (str): Nom du conteneur Azure.
        blob_service: Instance de BlobServiceClient.
        default: Valeur à retourner si le chargement échoue (None par défaut).

    Returns:
        Le modèle chargé ou la valeur par défaut en cas d'échec.
    """
    try:
        # Vérification de l'existence du blob
        blob_client = blob_service.get_blob_client(container=MODEL_CONTAINER, blob=blob_name)
        if not blob_client.exists():
            logger.warning(f"Blob {blob_name} introuvable dans le conteneur {MODEL_CONTAINER}")
            return default

        # Téléchargement et chargement
        download_stream = blob_client.download_blob()
        model = joblib.load(BytesIO(download_stream.readall()))
        logger.info(f"Modèle {blob_name} chargé avec succès")
        return model

    except ResourceNotFoundError:
        logger.error(f"Conteneur {MODEL_CONTAINER} ou blob {blob_name} introuvable")
        return default
    except PermissionError:
        logger.error(f"Permissions insuffisantes pour accéder à {blob_name}")
        return default
    except (EOFError, ValueError) as e:
        logger.error(f"Fichier {blob_name} corrompu: {str(e)}")
        return default
    except AzureError as e:
        logger.error(f"Erreur Azure lors du chargement de {blob_name}: {str(e)}")
        return default
    except Exception as e:
        logger.error(f"Erreur inattendue lors du chargement de {blob_name}: {str(e)}")
        return default


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
etat = 0
menu_container = ui.element('div')
modelblackdefense = load_model_from_blob("modelnoirdefense.pkl", "model", blob_service)
modelblackattaque = load_model_from_blob("modelnoirattaque.pkl", "model", blob_service)
modelblackdefensedames = load_model_from_blob("modelnoirdefensedames.pkl", "model", blob_service)
modelblackattaquedames = load_model_from_blob("modelnoirattaquedames.pkl", "model", blob_service)
modelblack = load_model_from_blob("modelnoirtransfert1.pkl", "model", blob_service)
modelblackdefensemaitre = load_model_from_blob("modelnoirdefense2.pkl", "model", blob_service)
modelblackattaquemaitre = load_model_from_blob("modelnoirattaque2.pkl", "model", blob_service)


dflist = load_model_from_blob("listes.pkl", "data", blob_service)
print("TF:", tf.__version__)
print("Keras:", keras.__version__)
print("Joblib:", joblib.__version__)
#modelwhite = joblib.load("modelblanctransfert2.pkl")
TAILLE = 10
tokens = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
newtokens = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
cells = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
state = {
    'rows': None,
    'cols': None,
    'rowclignotant': None,
    'colclignotant': None,
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
    'etat': 0,
    'mode': 0,
    'couleurpartie':None,
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
    dflist = load_model_from_blob("listes.pkl", "data", blob_service)
   
    # 🎯 FILTRAGE PAR MODE
    dflist = dflist[dflist['mode'] == state['mode']]

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

def retour_menu():
    # optionnel : remettre des états comme tu veux
    state['etat'] = 0      # ou 1 si tu préfères
    state['mode'] = None   # optionnel
    reset_game()
    game_container.set_visibility(False)
    menu_container.set_visibility(True)
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

def recherchedame(tokens):
    black = False
    white = False

    for ligne in tokens:
        for token in ligne:
            if token is not None:
                classes = token._classes

                if "dame" in classes:
                    if "black" in classes:
                        print('damenoir')
                        black = True
                    if "white" in classes:
                        print('dameblanche')
                        white = True

    return white, black

    return white, black
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
    mode_actuel = state['mode'] 
    TAILLE = 10
    tokens = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
    tokens = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
    cells = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]
    state = {
    'rows': None,
    'cols': None,
    'rowclignotant': None,
    'colclignotant': None,
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
    'couleurpartie':' ',
    'mode':mode_actuel,
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
    
def fin_de_partie(tokens, ndfdataframe, joueur, partiia):
    if partiia == 0:
            final_time = get_elapsed_time()  
    else:
            final_time = None
    ndfdataframe = ndfdataframe.copy()
    pions, pionsadverse = comptegame(tokens, joueur)
    if joueur == 'white':
        blancs = pions
        noirs = pionsadverse
    else:
        noirs = pions
        blancs = pionsadverse
        
    if blancs == 0 or state['gameover'] == 'white':
        if state['mode']== 4:
            reset_game()
            return 'noir', final_time
      
        ndfdataframe['vainqueur'] = ndfdataframe['joueur'].map({
    'black': 1,
    'white': 0
})
            
        df = load_model_from_blob("newdataiadames2.pkl", "data", blob_service)
        if df:
            max_partie = df['partie'].max()
            ndfdataframe['partie'] = max_partie + 1
            df = pd.concat(
            [df, ndfdataframe],
             ignore_index=True
             )
            save_model_to_blob(df, "newdataiadames2.pkl", "data", blob_service)
            reset_game()
            return 'noir', final_time

        else:
            ndfdataframe['partie'] = 1
            save_model_to_blob(df, "newdataiadames2.pkl", "data", blob_service)
            reset_game()
            return 'noir', final_time
    if noirs == 0 or state['gameover'] == 'black':
        if state['mode']== 4:
            reset_game()
            return 'blanc', final_time
        ndfdataframe['vainqueur'] = ndfdataframe['joueur'].map({
    'black': 1,
    'white': 0
})
        df = load_model_from_blob("newdataiadames2.pkl", "data", blob_service)
        if df:
            max_partie = df['partie'].max()
            ndfdataframe['partie'] = max_partie + 1
            df = pd.concat(
            [df, ndfdataframe],
             ignore_index=True
            )
            save_model_to_blob(df, "newdataiadames2.pkl", "data", blob_service)
            reset_game()
            return 'blanc', final_time

        else:
            ndfdataframe['partie'] = 1
            save_model_to_blob(ndfdataframe, "newdataiadames2.pkl", "data", blob_service)
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
        tens[dst_row][dst_col] = token

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

def max_captures_dame(row, col, board, couleur):
    captures = dame_captures_logic(row, col, board, couleur)
    if not captures:
        return 0

    best = 0

    for dst_row, dst_col, (mid_row, mid_col) in captures:
        new_board = copy_board(board)

        new_board[row][col] = None
        new_board[mid_row][mid_col] = None
        new_board[dst_row][dst_col] = ('dame', couleur)

        score = 1 + max_captures_dame(
            dst_row, dst_col, new_board, couleur
        )

        best = max(best, score)

    return best

def dame_captures_logic(row, col, board, couleur):
    """
    Retourne toutes les captures possibles pour une dame à (row, col).
    Format retour: [(dst_row, dst_col, (mid_row, mid_col)), ...]
    où (mid_row, mid_col) est la pièce ennemie capturée.
    """
    ennemi = 'white' if couleur == 'black' else 'black'
    captures = []

    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in directions:
        r = row + dr
        c = col + dc

        found_enemy = False
        enemy_pos = None

        # On avance sur la diagonale
        while 0 <= r < TAILLE and 0 <= c < TAILLE:
            cell = board[r][c]

            if cell is None:
                # Case vide
                if found_enemy:
                    # Si on a déjà vu 1 ennemi, toute case vide après est une destination possible
                    mid_row, mid_col = enemy_pos
                    captures.append((r, c, (mid_row, mid_col)))
                # sinon on continue à chercher un ennemi
                r += dr
                c += dc
                continue

            # Case occupée
            kind, color = cell

            if color == couleur:
                # Bloqué par une pièce amie
                break

            # Pièce ennemie
            if not found_enemy:
                found_enemy = True
                enemy_pos = (r, c)
                r += dr
                c += dc
                continue
            else:
                # Deux ennemis sur la même diagonale sans case vide entre → impossible
                break

    return captures

def max_captures_pion(row, col, board, couleur):
    captures = pion_captures_logic(row, col, board, couleur)

    if not captures:
        return 0

    best = 0

    for dst_row, dst_col, (mid_row, mid_col) in captures:
        # copie simple du plateau logique
        new_board = [r[:] for r in board]

        new_board[row][col] = None
        new_board[mid_row][mid_col] = None
        new_board[dst_row][dst_col] = ('pion', couleur)

        score = 1 + max_captures_pion(
            dst_row, dst_col, new_board, couleur
        )

        best = max(best, score)

    return best

def pion_captures_logic(row, col, board, couleur):
    captures = []

    if couleur == 'black':
        dir_row = 2
        ennemi = 'white'
    else:
        dir_row = -2
        ennemi = 'black'

    for dcol in (-2, 2):
        dst_row = row + dir_row
        dst_col = col + dcol

        if not (0 <= dst_row < TAILLE and 0 <= dst_col < TAILLE):
            continue

        mid_row = (row + dst_row) // 2
        mid_col = (col + dst_col) // 2

        if (
            board[mid_row][mid_col] == ('pion', ennemi)
            and board[dst_row][dst_col] is None
        ):
            captures.append((dst_row, dst_col, (mid_row, mid_col)))

    return captures

def max_captures_global(tokens, couleur):
    board = build_logic_board(tokens)
    max_global = 0

    for r in range(TAILLE):
        for c in range(TAILLE):
            piece = board[r][c]
            if not piece or piece[1] != couleur:
                continue

            kind, _ = piece

            if kind == 'pion':
                score = max_captures_pion(r, c, board, couleur)
            else:
                score = max_captures_dame(r, c, board, couleur)

            max_global = max(max_global, score)

    return max_global

def build_logic_board(tokens):
    board = [[None for _ in range(TAILLE)] for _ in range(TAILLE)]

    for r in range(TAILLE):
        for c in range(TAILLE):
            token = tokens[r][c]
            if token is None:
                continue

            classes = token._classes  # set de classes

            # couleur
            if 'black' in classes:
                couleur = 'black'
            elif 'white' in classes:
                couleur = 'white'
            else:
                continue  # pas une piece reconnue

            # type de piece
            if 'dame' in classes:
                kind = 'dame'
            elif 'token' in classes:
                kind = 'pion'
            else:
                continue

            board[r][c] = (kind, couleur)

    return board

def copy_board(board):
    return [row[:] for row in board]


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


def looksefairemanger(color, colormange, tab):
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
                   tab[0].append(mid_row)
                   tab[1].append(mid_col1)
                   
                    
                    

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
                        tab[0].append(mid_row)
                        tab[1].append(mid_col2)
                    
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
                                 tab[0].append(mid_row)
                                 tab[1].append(mid_col)
                             
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
                                 tab[0].append(mid_row)
                                 tab[1].append(mid_col)
                             
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
                             tab[0].append(mid_row)
                             tab[1].append(mid_col)
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
                             tab[0].append(mid_row)
                             tab[1].append(mid_col)
                                      
                s = s + 1

    return tab

def partiia():
    NB_PARTIES = 500

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
              move_black_ai_deep_learningia(joueur, adversaire)
        state['ndf']['dataaprescoup'] = data(tokens)
        global ndfdataframe
        ndfdataframe = pd.concat(
    [ndfdataframe, state['ndf'].to_frame().T],
    ignore_index=True
)
        state['ndf'] = pd.Series(dtype=object)
        resultat, timefinal = fin_de_partie(tokens, ndfdataframe, joueur, 1)
        print(resultat)
        if resultat == 'noir':
           ndfdataframe = pd.DataFrame()
           show_message('Victoire des noirs 🖤')
           break
        elif resultat == 'blanc':
           show_message('Victoire des blancs 🤍') 
           ndfdataframe = pd.DataFrame()
           break
        if state['coup'] > 250:
           ndfdataframe = pd.DataFrame()
           break
        if joueur == 'black':
           joueur='white'
           adversaire = 'black'
        else:
           joueur = 'black'
           adversaire = 'white'
        


def clicmode(r, c):
    if state['mode'] == 4:
        partiadeux(r, c)
    else:
        cell_clic(r, c)
    
def partiadeux(r, c):
    if state['rowclignotant']:
       rowc = int(state['rowclignotant'])  
       colc = int(state['colclignotant'])  
       if tokens[rowc][colc] is not None: 
             tokens[rowc][colc].classes(remove="anim")
             tokens[rowc][colc].classes(remove="animnoir")
    if state['couleurpartie'] =='black':
        couleur = 'black'
        couleurenemy = 'white'
    else:
        couleur = 'white'
        couleurenemy = 'black'
    global start_time, timer_started
    if not timer_started:
        start_time = time.time()
        timer_started = True
    lookeating(couleur, couleurenemy, state['tabeating'])
    if state['caseactive'] == False:
        selectcase(r, c, couleurenemy, couleur)
        return
    if state['caseactive'] == True:
        state['rowsaut'] = ' '
        state['colsaut'] = ' '
        if state['eatsaut'] != 1:
            state['coup'] += 1
            state['ndf']['coup'] = state['coup']
            state['ndf']['dataencour'] = data(tokens)
            state['ndf']['joueur'] = 'blanc'
            noirs, blancs = comptegame(tokens, couleur)
            state['ndf']['poslegal'], state['ndf']['deslegal'] = deplacements_possibles(tokens, couleur)
            if uniquement_false(state['ndf']['deslegal']) and state['eat'] != 1:
              state['gameover'] = couleur
            state['ndf']['nbpion'] = blancs
            state['ndf']['nbpionsadverse'] = noirs
        if state['gameover'] != couleur:
            movejetons(r, c, tokens, couleur, couleurenemy)
        state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
        lookeating(couleur, couleurenemy, state['tabeating'])
        state['eatsaut'] = 0
        for row, ligne in enumerate(state['tabeating']):
           for col, cell in enumerate(ligne):
              if cell.strip() != '':
                   if row == state['rowsaut'] and col == state['colsaut'] and 'select' in cell:
                       if state['transformation'] == 0:
                           print('eatsaut')
                           state['eatsaut'] = 1
            
        
    if state['caseactive'] == True and (state['eatsaut'] == 0 or state['transformation'] == 1):
        state['ndf']['poslegal'], state['ndf']['deslegal'] = deplacements_possibles(tokens, couleurenemy)
        if uniquement_false(state['ndf']['deslegal']):
             state['gameover'] = couleurenemy
             
        state['ndf']['poslegal'], state['ndf']['deslegal'] = deplacements_possibles(tokens, couleur)
        if uniquement_false(state['ndf']['deslegal']):
             state['gameover'] = couleur
        noirs, blancs = comptegame(tokens, couleurenemy)
        ndfdataframe = pd.DataFrame()
        resultat, timefinal = fin_de_partie(tokens, ndfdataframe, couleurenemy, 0)
        state['timefinal'] = timefinal
        if resultat == 'noir':
            ndfdataframe = pd.DataFrame()
            show_message(f'Victoire des noirs 🖤 en {timefinal}') 
            name_dialog.open()
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
        if state['couleurpartie'] == 'black':
          state['couleurpartie'] = 'white'
          show_message(f'C est le tour des blanc') 
        else:
          state['couleurpartie'] = 'black'
          show_message(f'C est le tour des noir') 




def cell_clic(r, c):
    global start_time, timer_started
    
    
    if state['rowclignotant']:
       rowc = int(state['rowclignotant'])  
       colc = int(state['colclignotant'])  
       if tokens[rowc][colc] is not None: 
             tokens[rowc][colc].classes(remove="anim")
             tokens[rowc][colc].classes(remove="animnoir")
       
    if not timer_started:
        start_time = time.time()
        timer_started = True
    lookeating('white', 'black', state['tabeating'])
    if state['caseactive'] == False:
        selectcase(r, c, 'black', 'white')
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
        if state['gameover'] != 'white':
            movejetons(r, c, tokens, 'white', 'black')
    state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
    lookeating('white', 'black', state['tabeating'])
    state['eatsaut'] = 0
    for row, ligne in enumerate(state['tabeating']):
           for col, cell in enumerate(ligne):
              if cell.strip() != '':
                   if row == state['rowsaut'] and col == state['colsaut'] and 'select' in cell:
                       if state['transformation'] == 0:
                           state['eatsaut'] = 1
    
    if state['eatsaut'] == 1 :
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
             
        state['ndf']['poslegal'], state['ndf']['deslegal'] = deplacements_possibles(tokens, 'white')
        if uniquement_false(state['ndf']['deslegal']):
             state['gameover'] = 'white'
        noirs, blancs = comptegame(tokens, 'black')
        if noirs != 0 and state['gameover'] != 'black' and state['gameover'] != 'white':
                 move_black_ai_deep_learning('black', 'white')
                 state['ndf']['poslegal'], state['ndf']['deslegal'] = deplacements_possibles(tokens, 'white')
                 if uniquement_false(state['ndf']['deslegal']):
                       state['gameover'] = 'white'
                 state['ndf']['dataaprescoup'] = data(tokens)
                 ndfdataframe = pd.concat(
                 [ndfdataframe, state['ndf'].to_frame().T],
                 ignore_index=True
)
        state['ndf'] = pd.Series(dtype=object)
    resultat, timefinal = fin_de_partie(tokens, ndfdataframe, 'black', 0)
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
def selectcase(row, col, couleurenemy, couleur):
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
                   
      if couleurenemy in token._classes:
        state['rows'] = None
        state['cols'] = None
        state['caseactive'] = False
        state['type'] = None
        state['eat'] = None
        state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
        show_message(f'Vous etes les {couleur} !')
        return
      if 'dame' in token._classes:
          state['type'] = 'dame'
      else:
          state['type'] = 'pion'
      state['rows'] = row 
      state['cols'] = col
      state['rowclignotant'] = row 
      state['colclignotant'] = col
      state['caseactive'] = True
      if couleur == 'white':
          print('couleur')
          print(couleur)
          tokens[state['rowclignotant']][state['colclignotant']].classes(add="anim")
      if couleur == 'black':
          print('couleur')
          print(couleur)
          tokens[state['rowclignotant']][state['colclignotant']].classes(add="animnoir")


def movejetons(row, col, tab, couleur, couleurenemy):
    state['transformation'] = 0

    if tab[row][col] is not None and state['eat'] == 0:
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
   
        
    if token is None :
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
                if mid_token is not None and couleurenemy in mid_token._classes:
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
               if mid_token is not None and couleurenemy in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None
                    manger = 1
                    state['rowsaut'] = row
                    state['colsaut'] = col
            
            if 'prise dame' in state['tabeating'][row][col] and ('basdroite' in state['tabeating'][src_row][src_col] and 'basdroite' in state['tabeating'][row][col]):
               mid_row = row - 1
               mid_col = col - 1
               mid_token = tab[mid_row][mid_col] 
               if mid_token is not None and couleurenemy in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None 
                    manger = 1
                    state['rowsaut'] = row
                    state['colsaut'] = col
               
            if 'prise dame' in state['tabeating'][row][col] and ('hautdroite' in state['tabeating'][src_row][src_col] and 'hautdroite' in state['tabeating'][row][col]):
               mid_row = row + 1
               mid_col = col - 1
               mid_token = tab[mid_row][mid_col] 
               if mid_token is not None and couleurenemy in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None  
                    manger = 1
                    state['rowsaut'] = row
                    state['colsaut'] = col
                
            if 'prise dame' in state['tabeating'][row][col] and ('hautgauche' in state['tabeating'][src_row][src_col] and 'hautgauche' in state['tabeating'][row][col]):

                mid_row = row + 1
                mid_col = col + 1
                mid_token = tab[mid_row][mid_col]
                if mid_token is not None and couleurenemy in mid_token._classes:
                    mid_token.delete()
                    tab[mid_row][mid_col] = None
                    manger = 1
                    state['rowsaut'] = row
                    state['colsaut'] = col
           
            if manger == 0:
                show_message('Vous devez obligatoirement manger !')
                state['rows'] = None
                state['cols'] = None
                state['caseactive'] = False
                state['type'] = None
                state['eat'] = None
                state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
                return 

            if couleur == 'white':
              dameblanche = creer_dame_blanche()
              dameblanche.classes(remove='selected')
              # retirer le jeton de l’ancienne case
              tab[src_row][src_col] = None
              cells[src_row][src_col].clear()
              # déplacer visuellement le jeton
              dameblanche.move(cells[row][col])
              tab[row][col] = dameblanche   
            else:
              damenoir = creer_dame_noire()
              damenoir.classes(remove='selected')
              # retirer le jeton de l’ancienne case
              tab[src_row][src_col] = None
              cells[src_row][src_col].clear()
              # déplacer visuellement le jeton
              damenoir.move(cells[row][col])
              tab[row][col] = damenoir    
        
        
                  
    if typejeton == 'pion' and state['eat'] == 0:  
      if couleur == 'white':
          deplacement = - 1
      else:
          deplacement =  1
      if (row == src_row + deplacement) and (col == src_col + 1 or col == src_col - 1):
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
            if couleur == 'white':
              # retirer la classe de sélection de l’ancienne position
              
              dameblanche = creer_dame_blanche()
              dameblanche.classes(remove='selected')
                
              

              dameblanche.move(cells[row][col])
              
              
              # retirer le jeton de l’ancienne case
              tab[src_row][src_col] = None
              cells[src_row][src_col].clear()
            
              
              tab[row][col] = dameblanche   
            else:
              # retirer la classe de sélection de l’ancienne position
              damenoir = creer_dame_noire()
              damenoir.classes(remove='selected')
              # retirer le jeton de l’ancienne case
             

              damenoir.move(cells[row][col])
             
              tab[src_row][src_col] = None
              cells[src_row][src_col].clear()
               # déplacer visuellement le jeton
              
              tab[row][col] = damenoir    
        else:
            show_message('Déplacement interdit !')
            state['rows'] = None
            state['cols'] = None
            state['caseactive'] = False
            state['type'] = None
            state['eat'] = None
            state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
            return  # impossible de se deplacer comme ça
                
                
   # transformation en dame
    if typejeton == 'pion':
      if couleur == 'white':
          if row == 0:
             print('transformationrow0')
             dameblanche = creer_dame_blanche()
             tab[row][col].delete()
             dameblanche.move(cells[row][col])
             tab[row][col] = dameblanche
             state['transformation'] = 1
      else:
          if row == 9:
             print('transformationrow9')
             damenoir = creer_dame_noire()
             tab[row][col].delete()
             damenoir.move(cells[row][col])
             tab[row][col] = damenoir
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
    global tokens
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
                board = build_logic_board(tokens)
                print("=== BOARD LOGIQUE ===")
                for r in range(TAILLE):
                   for c in range(TAILLE):
                      if board[r][c]:
                          print(r, c, board[r][c])
                max_local = max_captures_pion(src_row, src_col, board, couleur)
                max_global = max_captures_global(tokens, couleur)
                
                # ❌ ce pion ne fait pas le meilleur coup possible
                if max_local < max_global:
                   continue
                print('coupchoisi')
                print('max_localpion')
                print(max_local)
                print('max_global')
                print(max_global)
                print(src_row)
                print(src_col)
                # ✅ coup légal
                state['ndf']['type'] = 'pion'
                enchainement_pion(token, src_row, src_col, tokens, couleur)
                return
                

            if 'dame' in token._classes:
                 board = build_logic_board(tokens)
                 max_local = max_captures_dame(src_row, src_col, board, couleur)
                 max_global = max_captures_global(tokens, couleur)
                 if max_local < max_global:
                      continue  
                 print('coupchoisi')
                 print('max_localdame')
                 print(max_local)
                 print('max_global')
                 print(max_global)
                 print(src_row)
                 print(src_col)
                 state['ndf']['type'] = 'dame'
                 enchainement_dame(token, src_row, src_col, couleur)
                 return
                                   
    else:
      state['ndf']['typedeplacement'] = 'deplacement'
      boucle = 0
      newtokens = copy.deepcopy(tokens)
      coupinterdit = []
      while boucle < 3:
        state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
        table = [[], []]
        tabpionmanger = looksefairemanger('white', 'black', table)
        print('tabpionmanger')
        print(tabpionmanger)
        if any(tabpionmanger):
            boucle = 3
            print('captureavant')
        print(f'coupinterdit={coupinterdit}')
        if state['mode'] == 1:
           (row, col, new_row, new_col) = evaluationdeplacement(newtokens)
           sortideboucle = 1 
        else:
           (row, col, new_row, new_col), sortideboucle = evaluationdeplacement2model(newtokens, coupinterdit)
        token = tokens[row][col]
        tokens[row][col] = None
        token.move(cells[new_row][new_col])
        tokens[new_row][new_col] = token
        state['ndf']['row'] = new_row
        state['ndf']['col'] = new_col
        state['ndf']['srcrow'] = row
        state['ndf']['srccol'] = col
        state['ndf']['type'] = 'pion'
        state['tabeating'] = [[' ' for _ in range(TAILLE)] for _ in range(TAILLE)]
        table = [[], []]
        tabpionmanger = looksefairemanger('white', 'black', table)
        print('tabpionmanger')
        print(tabpionmanger)
        print(boucle)
        print(sortideboucle)
        if any(tabpionmanger) and boucle < 3 and sortideboucle == 0:
            print('verifsi nouvelle provoqué par le coup actif')
            print(new_row)
            print(new_col)
            print(tabpionmanger)
            print(len(tabpionmanger[0]))
            print(tabpionmanger[1][0])
            print(tabpionmanger[0][0])
            #suppressiondoublons
            coords = list(zip(tabpionmanger[0], tabpionmanger[1]))
            coords_uniques = list(dict.fromkeys(coords))
            tabpionmanger = [
    [r for r, c in coords_uniques],
    [c for r, c in coords_uniques],
]
            if (len(tabpionmanger[0]) == 1 and tabpionmanger[0][0] == new_row and tabpionmanger[1][0] == new_col):
                print('provoqué par le coup actif')
                break
            print('coup declenchant une capture passive')
            coupinterdit.append((row, col, new_row, new_col))
            print(coupinterdit)
            token = tokens[new_row][new_col]
            tokens[new_row][new_col] = None
            token.move(cells[row][col])
            tokens[row][col] = token
            
            
        else:
            break
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
                    score = modelblack.predict(patch_batch).item()
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
                        score = modelblack.predict(patch_batch).item()

                        
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

def evaluationdeplacement2model(tab, coupinterdit):

    meilleur_score = float('inf')
    meilleur_coup = None
    scores_coups = []  
    damesblanchepresente, damesnoirpresente  = recherchedame(tokens)
    if state['mode'] == 2:
       if damesblanchepresente:
                        print('damesblanchepresente')
                        nombrepatchdefense = 5
                        modelblackdefenseactif = modelblackdefensedames
       else:
                        print('damesblanchenonpresente')
                        nombrepatchdefense = 4
                        modelblackdefenseactif = modelblackdefense
                        
       if damesnoirpresente:
                        print('damesnoirpresente')
                        nombrepatchattaque = 5
                        modelblackattaqueactif = modelblackattaquedames
       else:
                        print('damesnoirnonpresente')
                        nombrepatchattaque = 4
                        modelblackattaqueactif = modelblackattaque 
    if state['mode'] == 3:
        nombrepatchdefense = 4
        nombrepatchattaque = 6
        modelblackdefenseactif = modelblackdefensemaitre
        modelblackattaqueactif = modelblackattaquemaitre
    table = [[], []]
    tabpionmanger = looksefairemanger('white', 'black', table)
    print(tabpionmanger)
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

                    if (row, col, new_row, new_col) in coupinterdit:
                        continue
                    # 🔹 COPIE LOGIQUE DU PLATEAU
                    tab_test = copy.deepcopy(tab)

                    # 🔹 SIMULATION LOGIQUE (PAS D'UI)
                    tab_test[row][col] = None
                    tab_test[new_row][new_col] = tab[row][col]

                    # 🔹 ENCODAGE POUR LE MODÈLE
                    plateau_avant = np.array(data(tab), dtype=np.float32)   # dataencour
                    plateau_apres = np.array(data(tab_test), dtype=np.float32)   # dataaprescou

                                            
                    patch = extract_patch(
                        plateau_apres,
                        int(new_row),
                        int(new_col),
                        k=nombrepatchdefense
                        )  # shape (5,5)

                    patch_batch = patch[np.newaxis, ..., np.newaxis]
                    
                    patch = extract_patch(
                        plateau_apres,
                        int(new_row),
                        int(new_col),
                        k=nombrepatchattaque
                        )  # shape (5,5)

                    patch_batchattaque = patch[np.newaxis, ..., np.newaxis]
                    
                    score = modelblackdefenseactif.predict(patch_batch).item()
                    scores_coups.append(
                    (score, (row, col, new_row, new_col), patch_batch, patch_batchattaque)
                    )
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
                        if (row, col, new_row, new_col) in coupinterdit:
                          new_row += dr
                          new_col += dc
                          continue
                        tab_test = copy.deepcopy(tab)
                        tab_test[row][col] = None
                        tab_test[new_row][new_col] = tab[row][col]
                        
                        plateau_avant = np.array(data(tab), dtype=np.float32)   # dataencour
                        plateau_apres = np.array(data(tab_test), dtype=np.float32)   # dataaprescoup

                        patch = extract_patch(
                        plateau_apres,
                        int(new_row),
                        int(new_col),
                        k=nombrepatchdefense
                        )  # shape (5,5)

                        patch_batch = patch[np.newaxis, ..., np.newaxis]

                        patch = extract_patch(
                        plateau_apres,
                        int(new_row),
                        int(new_col),
                        k=nombrepatchattaque
                        )  # shape (5,5)

                        patch_batchattaque = patch[np.newaxis, ..., np.newaxis]
                        score = modelblackdefenseactif.predict(patch_batch).item()
                        scores_coups.append(
                        (score, (row, col, new_row, new_col), patch_batch, patch_batchattaque)
                        )
                        
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

    # =========================
    # POST-SELECTION PAR BANDE + SECOND MODELE
    # =========================
    
    if len(scores_coups) == 0:
       
       return coupinterdit[0], 1  # sécurité

    #regarder s'il ya des piosn en dangers
    meilleurs_candidats = []
    if tabpionmanger:
      positions_autorisees = set(zip(tabpionmanger[0], tabpionmanger[1]))

      for score, (row, col, new_row, new_col), patch_batch, patchattaque in scores_coups:
        if (row, col) in positions_autorisees:
            print(row)
            print(col)
            print(score)

            # on stocke TOUTE la ligne scorecoup
            meilleurs_candidats.append(
                (score, (row, col, new_row, new_col), patch_batch, patchattaque)
            )

      # ⬇️ APRÈS la boucle : prendre le meilleur score
      newmeilleur_coup = None

      if meilleurs_candidats:
           newmeilleur_coup = meilleurs_candidats[0]

      for coup in meilleurs_candidats[1:]:
        if coup[0] > newmeilleur_coup[0]:
            newmeilleur_coup = coup

# résultat final
      if newmeilleur_coup:
           print("MEILLEUR COUP :", newmeilleur_coup)
           if newmeilleur_coup[0] < 0.7:
               (row, col, new_row, new_col) = newmeilleur_coup[1]
               newmeilleur_coup = (row, col, new_row, new_col)
               meilleur_coup = newmeilleur_coup
               return meilleur_coup, 0

    #regarder dans un premier temps si le meilleurs coup est secure sinon pas la peine de regarder le reste on prend le plus secure
    if meilleur_score > 0.5:
        return meilleur_coup, 0
    # on garde les coups dans [meilleur_score ; meilleur_score + 0.2]
    band = [
    (coup, patch, patchattaque)
    for score, coup, patch, patchattaque in scores_coups
      if score <= meilleur_score + 0.25
      ]

# sécurité
    if len(band) == 0:
      return meilleur_coup, 0
        
    print("meilleur_score =", meilleur_score)
    print("band (coups) =",
      [coup for coup, patch, patchattaque in band])
    scores_second_model = [
      modelblackattaqueactif.predict(patchattaque).item()
      for (_, patch, patchattaque) in band
]
    print("Scores du second modèle :")
    for i, (band_elem, score2) in enumerate(zip(band, scores_second_model)):
      coup = band_elem[0]   # ← uniquement le coup
      print(f"{i:02d} | coup = {coup} | score2 = {score2:.4f}")

# meilleur selon le second modèle
    best_idx = int(np.argmax(scores_second_model))
    meilleur_score = scores_second_model[best_idx]
    meilleur_coup = band[best_idx][0]

    print(best_idx)
    print('meilleur_coup')
    print(meilleur_coup)
    print(f"meilleur_score: {meilleur_score:.4f}")

# Sélectionner tous les coups avec une différence < 0.15
    meilleurs_candidats = [meilleur_coup]  # on commence avec le meilleur coup
    seuil = 0.05

    for i, (band_elem, score2) in enumerate(zip(band, scores_second_model)):
      if i == best_idx:  # on saute le meilleur coup (déjà ajouté)
        continue
    
      difference = abs(meilleur_score - score2)  # ← INDENTÉ + abs()
    
      if difference < seuil:
        coup = band_elem[0]
        meilleurs_candidats.append(coup)
        print(f"✓ Candidat retenu: {coup} | score = {score2:.4f} | diff = {difference:.4f}")
      else:
        print(f"✗ Rejeté: coup {band_elem[0]} | score = {score2:.4f} | diff = {difference:.4f}")

    print(f"\nNombre de candidats retenus: {len(meilleurs_candidats)}")
    print(f"Candidats: {meilleurs_candidats}")

# Choisir aléatoirement parmi les candidats
    coup_choisi = random.choice(meilleurs_candidats)
    print(f"Coup choisi aléatoirement: {coup_choisi}")

    return coup_choisi, 0




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


      
# valeur sélectionnée (DONNÉE SIMPLE)
choix_selectionne = {'value': None}

def choix_menu(valeur: str):
    choix_selectionne['value'] = valeur
    
def choisir_mode(valeur: str):
            state['etat'] = 1
            menu_container.set_visibility(False)
            print('bestresult')
            update_best_results() 
            game_container.set_visibility(True)


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
            
            global player_name
            if name_input.value:
                player_name = name_input.value
                print(state)
                players = pd.DataFrame(columns=['joueur', 'time', 'mode'])
                players.loc[len(players)] = {
                 'joueur': player_name,
                 'time': state['timefinal'],
                 'mode': state['mode']
                }
                
                dfliste = load_model_from_blob("listes.pkl", "data", blob_service)
                if dfliste:
                   dfliste = pd.concat(
                   [dfliste, players],
                   ignore_index=True
                   )
                  
                   save_model_to_blob(dflist, "listes.pkl", "data", blob_service)
                else:
                   save_model_to_blob(players, "listes.pkl", "data", blob_service)
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

ui.add_head_html("""
<style>
    /* Animation pour les cases blanches */
    @keyframes blink-white {
        0%, 100% { background-color: white; }
        50% { background-color: black; }
    }

    /* Animation pour les cases noires */
    @keyframes blink-brown {
        0%, 100% { background-color: #8B4513; } /* Marron (SaddleBrown) */
        50% { background-color: black; }
    }

    /* Classe générique pour le clignotement */
    .anim {
        animation: blink-white 2s infinite;
    }

    /* Classe spécifique pour les cases noires */
    .animnoir {
        animation: blink-brown 2s infinite;
    }
</style>
""")

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
    with ui.element('div').classes('center-column').style(
        'min-height: fit-content;'
    ):

        # 🎯 TITRE (TOUJOURS visible)
        ui.label('Jeu de dames').classes('title')

        # ================= MENU =================
        menu_container = ui.element('div')
        

        def choisir_mode(mode: int):
         state['mode'] = mode
         state['etat'] = 1
         menu_container.set_visibility(False)
         print('bestresult')
         update_best_results() 
         game_container.set_visibility(True)
         if mode == 4:
             show_message('C est le tour des blancs')
         

        with menu_container:
            ui.label('Choisissez votre adversaire').style(
        '''
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 16px;
        '''
    )
            with ui.column().style('gap: 12px; margin-top: 20px;'):
                for mode, texte, image_path in [
        (1, 'BOUBY (Facile)', '/static/images/enfant.png'),
        (2, 'Beth Harmon (Intermédiaire)', '/static/images/jeudeladame.png'),
        (3, 'Maitre (Difficile)', '/static/images/expert.png'),
        (4, 'Jouer contre un ami', '/static/images/expert.png'),           
    ]:
                    with ui.element('div').style(
                        '''
                       border: 2px solid black;
                       width: 220px;
                       height: 55px;

                       display: flex;
                       align-items: center;        
                       justify-content: flex-start;

                       padding-left: 20px;        
                       margin-left: 40px;         
                       margin-bottom: 10px;

                       cursor: pointer;
                       border-radius: 8px;
                       background-color: #e0e0e0;
                       font-weight: bold;
                        '''
                    ).on('click', lambda m=mode: choisir_mode(m)):
                        
                        ui.label(texte)

        # ================= JEU =================
        game_container = ui.element('div')
        game_container.set_visibility(False)

        with game_container:

            # ⏱️ TEMPS
            with ui.element('div').style(
    '''
    border: 2px solid black;
    padding: 6px 15px;
    margin: 0 auto 10px auto;  
    border-radius: 8px;
    background-color: #f5f5f5;
    font-size: 13px;
    width: fit-content;      
    '''
):
              time_label = ui.label('Temps : --:--')
            with ui.element('div').classes('board-layout'):
             # ✅ Bouton retour (tout en haut du jeu)
             with ui.row().style('width: 100%; justify-content: flex-start; margin-bottom: 10px;'):
                ui.button('⬅ Retour', on_click=retour_menu).props('outline')

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
                      cell.on('click', lambda r=row, c=col: (clicmode(r, c)))
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
   

ui.timer(1.0, update_time)

ui.run()
