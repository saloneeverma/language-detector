"""
Lexoglot — Language Family Classifier Training Script
======================================================
Trains a character n-gram TF-IDF + Logistic Regression model and exports
the weights to model_data.json, which can be embedded into lexoglot.html.

Usage:
    python train_lexoglot.py

To add more training data:
    1. Add sentences to TRAINING_DATA under the appropriate family key.
    2. Or point the script at external text files using load_external_data().
    3. Re-run the script — it will overwrite model_data.json.

To add a new language family:
    1. Add a new key to TRAINING_DATA with at least 20+ sample sentences.
    2. Add a corresponding entry to FAMILY_INFO in lexoglot.html.
    3. Re-run and re-embed the new model_data.json.

Requirements:
    pip install scikit-learn numpy
"""

import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# ── Training Data ────────────────────────────────────────────────────────────
# Each family needs at least ~20 samples for reasonable results.
# More is always better — aim for 200+ per family for strong invented-word accuracy.
# Samples should be romanized/transliterated for non-Latin scripts so letter
# patterns are preserved (e.g. "zdravstvuyte" not "здравствуйте").

TRAINING_DATA = {
    "Romance": [
        "le soleil brille sur la mer bleue","bonjour comment allez vous aujourd hui",
        "la maison est belle et grande","nous mangeons du pain et du fromage",
        "el cielo es azul y hermoso","buenos dias como estas hoy",
        "la casa tiene muchas ventanas","me gusta mucho la musica",
        "il sole splende sul mare","buongiorno come stai oggi",
        "la casa e bella e grande","mi piace molto la musica italiana",
        "o sol brilha sobre o mar azul","bom dia como voce esta",
        "a casa tem muitas janelas bonitas","gosto muito de musica portuguesa",
        "beau magnifique rouge blanc bleu","hermosa bonita rojo blanco azul",
        "bella bellissima rosso bianco blu","linda bonita vermelho branco azul",
        "amour coeur fleur jardin maison","amor corazon flor jardin casa",
        "amore cuore fiore giardino casa","amor coracao flor jardim casa",
        "chanter danser manger partir venir","cantar bailar comer salir venir",
        "cantare ballare mangiare uscire venire","cantar dançar comer sair vir",
        "nuit jour matin soir temps","noche dia manana tarde tiempo",
        "notte giorno mattina sera tempo","noite dia manha tarde tempo",
    ],
    "Germanic": [
        "the sun is shining on the blue sea","hello how are you doing today",
        "the house is beautiful and large","we are eating bread and cheese",
        "die sonne scheint auf das blaue meer","guten morgen wie geht es ihnen",
        "das haus ist schoen und gross","ich mag musik sehr gerne",
        "de zon schijnt op de blauwe zee","goedemorgen hoe gaat het met u",
        "het huis is mooi en groot","ik hou van muziek luisteren",
        "solen skiner over det blaa hav","god morgen hvordan har du det",
        "huset er smukt og stort","jeg elsker at lytte til musik",
        "beautiful wonderful red white blue","schoen wunderschoen rot weiss blau",
        "mooi prachtig rood wit blauw","smuk vidunderlig rod hvid blaa",
        "love heart flower garden home","liebe herz blume garten haus",
        "liefde hart bloem tuin huis","kaerlighed hjerte blomst have hjem",
        "strength knowledge freedom truth light","staerke wissen freiheit wahrheit licht",
        "night day morning evening time","nacht tag morgen abend zeit",
        "nacht dag morgen avond tijd","nat dag morgen aften tid",
        "sing dance eat leave come","singen tanzen essen gehen kommen",
        "zingen dansen eten weggaan komen","synge danse spise tage komme",
    ],
    "Slavic": [
        "solntse svetit nad sinem morem","dobroe utro kak dela segodnya",
        "dom krasivyi i bolshoy","my edim khleb i syr",
        "slonce swieci nad niebieskim morzem","dzien dobry jak sie masz",
        "dom jest piekny i duzy","lubie sluchac muzyki bardzo",
        "slunce sviti nad modrim morem","dobre rano jak se mas",
        "dum je krasny a velky","rad posloucham hudbu velmi",
        "krasivyi zamechatelnyi krasnyi belyi sinyi","piekny wspanialy czerwony bialy niebieski",
        "krasny uzasny cerveny bily modry",
        "lyubov serdtse tsvetok sad dom","milosc serce kwiat ogrod dom",
        "laska srdce kvet zahrada dum",
        "sila znanie svoboda pravda svet","sila wiedza wolnosc prawda swiatlo",
        "sila znalost svoboda pravda svetlo",
        "privet zdravstvuyte tovarishch drug brat","czesc dzien dobry kolega przyjaciel",
        "ahoj dobry den kamarad pritel",
        "noch den utro vecher vremya","noc dzien rano wieczor czas",
        "noc den rano vecer cas",
        "pes kocka ryba ptak strom","pies kot ryba ptak drzewo",
    ],
    "Semitic": [
        "shemesh zorahat al hayam hakachol","boker tov mah shlomcha hayom",
        "habayit yafe vegadol meod","anachnu ochlim lechem vegvinah",
        "alshams tushriqu ala albahr alazraq","sabah alkhayr kayfa haluka alyawm",
        "albaytu jamilun wakabir jidan","uhibbu almusiqaa kathiran jidan",
        "yafe nifla adom lavan kachol","jamil raai ahmar abyad azraq",
        "ahavah lev perach gan bayit","hubb qalb zahra hadiqah bayt",
        "koach daat cherut emet or","quwwah marifah hurriyyah haqiqah nur",
        "shalom ahlan marhaba salam aleikum","lehitraot shalom shabbat mazal tov",
        "layla yom boker erev zman","layl yawm sabah masa waqt",
        "shir rikud okhel yatza ba","ghinaa raqs akl khrj ata",
        "kelev chatul dag tsipor etz","kalb qitta samaka tayr shajara",
    ],
    "Turkic": [
        "gunes mavi denizin uzerinde parlıyor","gunaydin bugun nasılsınız",
        "ev cok guzel ve buyuk","ekmek ve peynir yiyoruz",
        "kun koktom ustunde parlaydy","kaiyrly tan bugun kalay",
        "guzel muhtesem kirmizi beyaz mavi","suluu ajayip kyzyl ak kok",
        "ask kalp cicek bahce ev","mahabbat zhürek gül bak üy",
        "guc bilgi ozgurluk gercek isik","kuch bilim erkinlik chyndyk nur",
        "merhaba selam arkadas kardes dost","salam dos brat dostor kandas",
        "yemek icmek gelmek gitmek olmak","baruu keluu ketuu boluu",
        "gece gun sabah aksam zaman","tun kun tan kech waqt",
        "kopeye kedi balik kus agac","it mushu balyk qush jaghach",
        "sevgi yurek cicek bahce ev","süyüü jürök gül bak üy",
        "cok güzel harika mükemmel enfes","jakshi zor ajayip ukuk",
    ],
    "Dravidian": [
        "suryan nila kadal mele olichukondu irukku","vanakkam inru eppadi irukkingal",
        "veedu romba azhagaana perithaana","nangal roti matrum paneer saapidugirorm",
        "surya nillu samudra meele prakaashikunnu","namaskaram innu enthaanu varthamaanam",
        "veedu valare sundharamaaya perithaaya","njangal rotti matrum paneer kazhikkunnu",
        "azhagu adbhutham sivappu veluppu nilam","sundharam ascharyam sivappu vellai neelam",
        "kadhal idhayam poo thottam veedu","prema hridayam pookkal thottam veedu",
        "shakti jnanam swaathanthryam sathyam prakasham",
        "vanakkam namaskaram ayya amma anna","namaskaaram ayye amma etta chechi",
        "thinnu kudicchu vannu ponu aahu","kazichchu kudichchu vannu poyi aayi",
        "iravil pagal kaalai maalai neram","raathri pagal raavile maalai samayam",
        "naai poonai meen paravai maram","nayi poocha meen pakshi maram",
        "paadu aadu saapu poo va","paadi aadi kazhichchu poo va",
    ],
    "Uralic": [
        "aurinko paistaa sinisen meren yllä","hyvää huomenta kuinka voitte tänään",
        "talo on kaunis ja suuri","me syomme leipaa ja juustoa",
        "a nap süt a kék tenger felett","jo reggelt hogy van ma",
        "a ház szép és nagy","mi kenyeret és sajtot eszünk",
        "kaunis ihana punainen valkoinen sininen","szep csodálatos piros feher kek",
        "rakkaus sydan kukka puutarha koti","szerelem szív virág kert otthon",
        "voima tieto vapaus totuus valo","ero tudasze szabadsag igazsag feny",
        "hei moi hyvää päivää terve","szia hello jó napot üdvözlöm",
        "tulla mennä syödä juoda olla","jonni menni enni inni lenni",
        "koira kissa lintu kala puu","kutya macska madár hal fa",
        "yö päivä aamu ilta aika","éjszaka nap reggel este idő",
        "laulaa tanssia syödä lähteä tulla","énekel táncol eszik megy jön",
        "yksi kaksi kolme nelja viisi","egy ketto harom negy ot",
    ],
    "Indo-Iranian": [
        "suraj neele sagar par chamak raha hai","namaste aaj aap kaise hain",
        "ghar bahut sundar aur bada hai","ham roti aur paneer khate hain",
        "khorshid bar darya abi mi tabad","sobh bekhayr emrooz hale shoma chetour ast",
        "khaneh besyar ziba va bozorg ast","ma nan va panir mikhorim",
        "sundar adbhut lal safed nila","ziba ajib qermez sefid abi",
        "prem dil phool bageecha ghar","eshq del gol bagh khaneh",
        "shakti gyan azaadi sachhai roshni","nirou danesh azadi haqiqat nur",
        "namaste namaskar bhai behan dost","dorud salam baradar khahar dust",
        "khana peena aana jaana hona","khordan ashidan amadan raftan budan",
        "raat din subah shaam waqt","shab ruz sobh asr vaqt",
        "kutta billi machli chidiya pedh","sag gorbeh mahi morgh darakht",
        "gaana naachna khaana jaana aana","khanandan raghsidan khordan raftan amadan",
    ],
}


def load_external_data(file_path: str, family: str, max_lines: int = 500) -> list[str]:
    """
    Load training sentences from a plain text file (one sentence per line).
    Useful for bulk-loading data from OPUS or other corpora.

    Args:
        file_path: Path to a .txt file with one sentence per line.
        family:    The language family label to assign to all lines.
        max_lines: Cap on how many lines to load (avoids memory issues).

    Returns:
        List of sentences (strings).

    Example:
        extra = load_external_data("data/french.txt", "Romance", max_lines=1000)
        TRAINING_DATA["Romance"].extend(extra)
    """
    sentences = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if line:
                sentences.append(line)
    print(f"  Loaded {len(sentences)} lines from {file_path} → {family}")
    return sentences


def build_training_arrays(data: dict) -> tuple[list, list]:
    """Flatten the TRAINING_DATA dict into parallel (texts, labels) lists."""
    texts, labels = [], []
    for family, samples in data.items():
        for s in samples:
            texts.append(s)
            labels.append(family)
    return texts, labels


def train(texts: list, labels: list) -> Pipeline:
    """
    Build and fit the classification pipeline.

    Model architecture:
      - TfidfVectorizer with char_wb analyzer: extracts character 2–4-grams
        from word-boundary-padded text, capturing both within-word patterns
        and word-edge patterns (very useful for distinguishing families).
      - sublinear_tf=True: applies log normalization to term frequencies,
        reducing the dominance of very common n-grams.
      - LogisticRegression with lbfgs solver: fast multinomial classifier,
        well-suited for high-dimensional sparse features.

    Tuning tips:
      - Increase max_features (default 8000) if you have more training data.
      - Increase C (default 5.0) for less regularization if data is abundant.
      - Try ngram_range=(2, 5) for longer-pattern sensitivity.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=8000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
        )),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def evaluate(pipeline: Pipeline, texts: list, labels: list) -> None:
    """Run 5-fold cross-validation and print per-family accuracy."""
    print("\n── Cross-validation (5-fold) ─────────────────────────")
    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
    print(f"  Mean accuracy : {scores.mean():.3f}")
    print(f"  Std deviation : {scores.std():.3f}")
    print(f"  Per-fold      : {[round(s, 3) for s in scores]}")


def smoke_test(pipeline: Pipeline) -> None:
    """Classify a set of known and invented words as a quick sanity check."""
    test_cases = [
        # (word, expected_family)
        ("bonjour",        "Romance"),
        ("schmetterling",  "Germanic"),
        ("zdravstvuyte",   "Slavic"),
        ("marhaba",        "Semitic"),
        ("gunaydin",       "Turkic"),
        ("vanakkam",       "Dravidian"),
        ("aurinko",        "Uralic"),
        ("namaste",        "Indo-Iranian"),
        # Invented words — no guaranteed answer, just showing the model works
        ("kvelthrion",     "?"),
        ("azumar",         "?"),
        ("throknul",       "?"),
    ]
    print("\n── Smoke test ────────────────────────────────────────")
    classes = pipeline.classes_
    for word, expected in test_cases:
        pred = pipeline.predict([word])[0]
        proba = pipeline.predict_proba([word])[0]
        top3 = sorted(zip(classes, proba), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{c}: {p:.2f}" for c, p in top3)
        match = "✓" if pred == expected else ("~" if expected == "?" else "✗")
        print(f"  {match} {word:20s} → {pred:15s} | {top3_str}")


def export_model(pipeline: Pipeline, output_path: str) -> None:
    """
    Serialize model weights to JSON for embedding in lexoglot.html.

    The exported file contains:
      vocab      — {ngram: feature_index} lookup table
      idf        — IDF weights (one per feature)
      weights    — classifier weight matrix (n_classes × n_features)
      intercept  — classifier bias terms (one per class)
      classes    — ordered list of class labels

    The JavaScript inference in lexoglot.html mirrors this exactly:
      1. Extract char n-grams from the input text
      2. Apply sublinear TF + IDF weighting
      3. L2-normalize the feature vector
      4. Dot-product with weights + intercept → logits
      5. Softmax → per-family probabilities
    """
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    model_data = {
        "vocab":     {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "idf":       [float(x) for x in vectorizer.idf_],
        "weights":   [[float(x) for x in row] for row in clf.coef_],
        "intercept": [float(x) for x in clf.intercept_],
        "classes":   [str(x) for x in clf.classes_],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n── Export ────────────────────────────────────────────")
    print(f"  Saved to      : {output_path}")
    print(f"  File size     : {size_kb:.1f} KB")
    print(f"  Vocab size    : {len(model_data['vocab'])} n-grams")
    print(f"  Classes       : {model_data['classes']}")
    print(f"\n  To embed in lexoglot.html, replace the MODEL = {{...}} value")
    print(f"  with the contents of {output_path}.")


def embed_into_html(model_path: str, html_path: str) -> None:
    """
    Convenience function: read model_data.json and patch it directly
    into an existing lexoglot.html, replacing the MODEL = {...} assignment.

    Args:
        model_path: Path to the exported model_data.json.
        html_path:  Path to lexoglot.html (will be overwritten in-place).
    """
    import re
    with open(model_path, encoding="utf-8") as f:
        model_json = f.read()
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    # Replace const MODEL = {...}; with fresh weights
    new_html = re.sub(
        r"const MODEL = \{.*?\};",
        f"const MODEL = {model_json};",
        html,
        flags=re.DOTALL,
    )
    if new_html == html:
        print("  Warning: could not find 'const MODEL = {...};' in the HTML. "
              "Paste model_data.json manually.")
    else:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(new_html)
        print(f"  Patched model into {html_path} successfully.")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Lexoglot — Training Language Family Classifier")
    print("=" * 52)

    # ── Optional: load external data files ──────────────────
    # Uncomment and adjust paths to bulk-load data from OPUS or similar:
    #
    # extra_french = load_external_data("data/french.txt", "Romance", max_lines=500)
    # TRAINING_DATA["Romance"].extend(extra_french)
    #
    # extra_russian = load_external_data("data/russian_romanized.txt", "Slavic")
    # TRAINING_DATA["Slavic"].extend(extra_russian)

    texts, labels = build_training_arrays(TRAINING_DATA)
    counts = {fam: labels.count(fam) for fam in set(labels)}
    print(f"\nTraining samples : {len(texts)} total")
    for fam, n in sorted(counts.items()):
        print(f"  {fam:20s} {n} samples")

    print("\nFitting model...")
    pipeline = train(texts, labels)
    print("Done.")

    evaluate(pipeline, texts, labels)
    smoke_test(pipeline)

    output_path = "model_data.json"
    export_model(pipeline, output_path)

    # ── Optional: auto-patch lexoglot.html ──────────────────
    # If lexoglot.html is in the same directory, uncomment this to
    # update it automatically every time you retrain:
    #
    # html_path = "lexoglot.html"
    # if os.path.exists(html_path):
    #     embed_into_html(output_path, html_path)