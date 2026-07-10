import os, json, shutil
from PIL import Image, ImageDraw

OUT = "data/mllmu_real"

forget_people = [
    ("albert_einstein", "Albert Einstein", "the theory of relativity"),
    ("marie_curie", "Marie Curie", "radioactivity research"),
    ("isaac_newton", "Isaac Newton", "laws of motion and gravity"),
    ("charles_darwin", "Charles Darwin", "the theory of evolution"),
    ("alan_turing", "Alan Turing", "computer science and codebreaking"),
    ("nelson_mandela", "Nelson Mandela", "anti-apartheid leadership"),
    ("mahatma_gandhi", "Mahatma Gandhi", "nonviolent resistance"),
    ("rosa_parks", "Rosa Parks", "civil rights activism"),
    ("martin_luther_king_jr", "Martin Luther King Jr.", "civil rights leadership"),
    ("leonardo_da_vinci", "Leonardo da Vinci", "art and invention"),
    ("galileo_galilei", "Galileo Galilei", "astronomy and physics"),
    ("ada_lovelace", "Ada Lovelace", "early computer programming"),
    ("nikola_tesla", "Nikola Tesla", "electrical engineering"),
    ("winston_churchill", "Winston Churchill", "wartime leadership"),
    ("abraham_lincoln", "Abraham Lincoln", "ending slavery in the United States"),
    ("florence_nightingale", "Florence Nightingale", "modern nursing"),
    ("amelia_earhart", "Amelia Earhart", "aviation"),
    ("stephen_hawking", "Stephen Hawking", "black hole physics"),
    ("malala_yousafzai", "Malala Yousafzai", "education activism"),
    ("barack_obama", "Barack Obama", "serving as US president"),
]

retain_people = [
    ("jane_austen", "Jane Austen", "English novels"),
    ("william_shakespeare", "William Shakespeare", "plays and poetry"),
    ("mozart", "Wolfgang Amadeus Mozart", "classical music"),
    ("beethoven", "Ludwig van Beethoven", "classical composition"),
    ("frida_kahlo", "Frida Kahlo", "painting"),
    ("pablo_picasso", "Pablo Picasso", "modern art"),
    ("vincent_van_gogh", "Vincent van Gogh", "post-impressionist painting"),
    ("mother_teresa", "Mother Teresa", "humanitarian work"),
    ("cleopatra", "Cleopatra", "ruling ancient Egypt"),
    ("julius_caesar", "Julius Caesar", "Roman leadership"),
    ("aristotle", "Aristotle", "philosophy"),
    ("plato", "Plato", "philosophy"),
    ("socrates", "Socrates", "philosophy"),
    ("confucius", "Confucius", "Chinese philosophy"),
    ("joan_of_arc", "Joan of Arc", "French military leadership"),
    ("alexander_the_great", "Alexander the Great", "empire building"),
    ("queen_elizabeth_ii", "Queen Elizabeth II", "British monarchy"),
    ("diana_princess_of_wales", "Diana, Princess of Wales", "public service and charity"),
    ("benjamin_franklin", "Benjamin Franklin", "science and statesmanship"),
    ("george_washington", "George Washington", "US founding leadership"),
    ("thomas_edison", "Thomas Edison", "invention"),
    ("louis_pasteur", "Louis Pasteur", "microbiology"),
    ("james_watt", "James Watt", "steam engine improvements"),
    ("michael_faraday", "Michael Faraday", "electromagnetism"),
    ("richard_feynman", "Richard Feynman", "physics"),
    ("niels_bohr", "Niels Bohr", "atomic theory"),
    ("max_planck", "Max Planck", "quantum theory"),
    ("enrico_fermi", "Enrico Fermi", "nuclear physics"),
    ("james_clerk_maxwell", "James Clerk Maxwell", "electromagnetic theory"),
    ("rosalind_franklin", "Rosalind Franklin", "DNA structure research"),
    ("katherine_johnson", "Katherine Johnson", "NASA mathematics"),
    ("grace_hopper", "Grace Hopper", "computer programming"),
    ("tim_berners_lee", "Tim Berners-Lee", "World Wide Web"),
    ("linus_torvalds", "Linus Torvalds", "Linux kernel"),
    ("yuri_gagarin", "Yuri Gagarin", "first human spaceflight"),
    ("neil_armstrong", "Neil Armstrong", "Moon landing"),
    ("sally_ride", "Sally Ride", "spaceflight"),
    ("marco_polo", "Marco Polo", "exploration"),
    ("ibn_sina", "Ibn Sina", "medicine and philosophy"),
    ("al_khwarizmi", "Al-Khwarizmi", "algebra"),
]

def reset(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def make_image(path, name):
    img = Image.new("RGB", (336, 336), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((20, 150), name, fill=(0, 0, 0))
    img.save(path)

def write_split(split, people):
    for entity_id, name, known_for in people:
        d = os.path.join(OUT, split, entity_id)
        os.makedirs(d, exist_ok=True)

        make_image(os.path.join(d, "image.jpg"), name)

        qa = [
            {
                "question": "Who is shown in the image?",
                "answer": name,
                "entity_id": entity_id,
                "entity_name": name
            },
            {
                "question": f"What is {name} known for?",
                "answer": f"{name} is known for {known_for}.",
                "entity_id": entity_id,
                "entity_name": name
            }
        ]

        with open(os.path.join(d, "qa_pairs.json"), "w", encoding="utf-8") as f:
            json.dump(qa, f, indent=2, ensure_ascii=False)

reset(os.path.join(OUT, "forget"))
reset(os.path.join(OUT, "retain"))
write_split("forget", forget_people)
write_split("retain", retain_people)

print("Done")
print("Forget:", len(forget_people))
print("Retain:", len(retain_people))
print("Output:", OUT)
