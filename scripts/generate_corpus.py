#!/usr/bin/env python
"""Generates the bundled document corpus used by this project.

The corpus is SYNTHETIC: it is built from hand-written fact banks organized
by domain -> subtopic, then assembled into short "articles" by sampling a
handful of sentences per subtopic. This is documented honestly in the README
-- we did not scrape or license a real news/article dataset. The point of the
exercise is to have a corpus with genuine, distinguishable topical structure
(so BM25/embeddings/eval numbers are real and reproducible), not to claim
these are real articles.

Run:
    python scripts/generate_corpus.py
Writes data/corpus.jsonl
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.config import CORPUS_PATH  # noqa: E402
from semantic_search.corpus import Document, write_corpus  # noqa: E402

RANDOM_SEED = 42
DOCS_PER_SUBTOPIC = 20

# domain -> subtopic -> (title_template, [fact sentences])
FACT_BANKS: dict[str, dict[str, tuple[str, list[str]]]] = {
    "technology": {
        "cloud-computing": (
            "Cloud Computing",
            [
                "Cloud computing lets organizations rent compute, storage, and networking from a provider instead of owning hardware.",
                "Elastic autoscaling adjusts the number of running instances based on real-time traffic and load metrics.",
                "Multi-tenant data centers isolate customer workloads using virtualization and container runtimes.",
                "Object storage services like S3-compatible buckets are designed for durability across multiple availability zones.",
                "Serverless functions bill by execution time and memory rather than by provisioned server hours.",
                "Infrastructure as code tools such as Terraform describe cloud resources declaratively in version-controlled files.",
                "A well-designed cloud architecture separates stateless compute from durable, replicated storage layers.",
                "Cold starts in serverless platforms occur when a new execution environment must be initialized before a request is served.",
                "Content delivery networks cache static assets at edge locations close to end users to cut latency.",
                "Cloud cost optimization often focuses on right-sizing instances and eliminating idle reserved capacity.",
                "Managed database services handle patching, backups, and failover automatically for the customer.",
                "Hybrid cloud deployments keep sensitive workloads on-premises while bursting extra capacity to the public cloud.",
                "Identity and access management policies define which cloud principals can invoke which resources.",
                "Region and availability-zone placement decisions balance latency, compliance, and disaster-recovery requirements.",
                "Cloud providers publish service level agreements that specify uptime guarantees and financial credits for outages.",
                "Container orchestration platforms schedule workloads across a cluster of virtual machines automatically.",
                "Spot or preemptible instances offer discounted compute in exchange for the possibility of sudden termination.",
                "Observability stacks in the cloud combine metrics, logs, and distributed traces to diagnose incidents.",
            ],
        ),
        "machine-learning": (
            "Machine Learning",
            [
                "Machine learning models learn statistical patterns from labeled or unlabeled training data.",
                "Gradient descent iteratively adjusts model weights to minimize a differentiable loss function.",
                "Overfitting happens when a model memorizes training examples instead of learning generalizable patterns.",
                "Cross-validation splits data into folds to estimate how a model will perform on unseen examples.",
                "Feature engineering transforms raw inputs into representations that make patterns easier for a model to learn.",
                "Neural networks stack layers of weighted connections and nonlinear activation functions.",
                "Transfer learning reuses weights from a model pretrained on a large dataset to speed up training on a smaller one.",
                "Regularization techniques like dropout and weight decay discourage a model from fitting noise in the data.",
                "Hyperparameter tuning searches over settings such as learning rate and batch size to improve validation performance.",
                "Ensemble methods combine multiple weaker models to produce a single, more accurate prediction.",
                "Embeddings map discrete items like words or documents into dense vectors that capture semantic similarity.",
                "Attention mechanisms let a model weigh the relevance of different input tokens when producing an output.",
                "Precision and recall trade off false positives against false negatives when evaluating a classifier.",
                "Batch normalization stabilizes training by rescaling activations within each mini-batch.",
                "Reinforcement learning trains an agent to maximize cumulative reward through trial-and-error interaction with an environment.",
                "A confusion matrix summarizes how a classifier's predictions compare against ground-truth labels.",
                "Data augmentation artificially expands a training set by applying label-preserving transformations to existing examples.",
                "Model interpretability techniques like SHAP values attribute a prediction to individual input features.",
            ],
        ),
        "cybersecurity": (
            "Cybersecurity",
            [
                "Phishing attacks trick users into revealing credentials through deceptive emails or websites.",
                "Multi-factor authentication requires a second proof of identity beyond a password, such as a one-time code.",
                "Encryption at rest protects stored data even if the underlying storage medium is stolen.",
                "A zero-day vulnerability is a flaw that is exploited before the vendor has released a patch.",
                "Firewalls filter network traffic based on rules that allow or deny packets by port, protocol, or address.",
                "Ransomware encrypts a victim's files and demands payment in exchange for the decryption key.",
                "Penetration testing simulates real attacks against a system to uncover exploitable weaknesses.",
                "Least-privilege access control limits each account to only the permissions it strictly needs.",
                "Security information and event management systems aggregate logs to detect suspicious activity in real time.",
                "A denial-of-service attack overwhelms a target with traffic so legitimate users cannot get through.",
                "Patch management processes track and apply vendor security updates across an organization's fleet.",
                "Public key infrastructure issues digital certificates that bind a public key to a verified identity.",
                "Social engineering exploits human trust rather than technical flaws to gain unauthorized access.",
                "Endpoint detection and response tools monitor devices for malicious behavior after initial compromise.",
                "A security incident response plan defines the steps a team takes to contain and recover from a breach.",
                "Threat intelligence feeds share indicators of compromise observed across many organizations.",
                "Data loss prevention systems block sensitive information from leaving an organization's network.",
                "Supply-chain attacks compromise a trusted vendor's software to reach many downstream victims at once.",
            ],
        ),
        "mobile-apps": (
            "Mobile App Development",
            [
                "Native mobile apps are built with platform-specific SDKs like Swift for iOS or Kotlin for Android.",
                "Cross-platform frameworks such as React Native share a single codebase across iOS and Android.",
                "Push notifications let an app re-engage users even when the app is not in the foreground.",
                "App store review guidelines govern what content and behavior are permitted before publication.",
                "Offline-first design caches data locally so an app remains usable without a network connection.",
                "Battery-efficient background tasks are scheduled to run during low-power windows on mobile devices.",
                "Responsive layouts adapt an app's interface to different screen sizes and orientations.",
                "In-app purchases let developers monetize through consumable, subscription, or one-time unlockable content.",
                "Mobile analytics SDKs track screen views, taps, and funnels to understand user behavior.",
                "Deep links open a specific screen inside an app directly from a URL or notification.",
                "App performance profiling identifies slow frame renders and memory leaks on real devices.",
                "Biometric authentication uses a fingerprint or face scan to unlock sensitive app features.",
                "Over-the-air updates let some app logic change without requiring a new app store release.",
                "Accessibility APIs expose screen-reader labels so visually impaired users can navigate an app.",
                "Mobile app crash reporting tools capture stack traces from real user sessions in production.",
                "Local notification scheduling lets an app remind a user at a specific time without a server round trip.",
                "App size optimization strips unused code and compresses assets to reduce download size.",
                "Continuous delivery pipelines for mobile automate build, signing, and distribution to test devices.",
            ],
        ),
        "databases": (
            "Databases",
            [
                "Relational databases organize data into tables linked by foreign keys and enforce a fixed schema.",
                "Indexes speed up query lookups at the cost of extra storage and slower writes.",
                "ACID transactions guarantee atomicity, consistency, isolation, and durability for a set of operations.",
                "NoSQL document stores trade strict schemas for flexible, nested JSON-like records.",
                "Database sharding splits a large table across multiple machines to scale writes horizontally.",
                "Replication copies data to secondary nodes to improve read throughput and provide failover.",
                "Query planners choose an execution strategy by estimating the cost of alternative join orders.",
                "Normalization reduces redundancy by splitting data into related tables, while denormalization trades that off for read speed.",
                "Vector databases store high-dimensional embeddings and support approximate nearest-neighbor search.",
                "Write-ahead logging records changes before they are applied so a database can recover after a crash.",
                "Connection pooling reuses database connections across requests to avoid expensive setup costs.",
                "Optimistic concurrency control detects conflicting updates by checking a row's version before committing.",
                "Time-series databases are optimized for append-heavy workloads keyed by timestamp.",
                "A materialized view precomputes and stores the result of an expensive query for fast repeated reads.",
                "Database migrations version schema changes so they can be applied consistently across environments.",
                "Full-text search indexes tokenize and rank documents by relevance to a query string.",
                "Isolation levels like serializable or read-committed control how transactions see each other's uncommitted changes.",
                "Backup and point-in-time recovery let a database be restored to a specific moment before data loss.",
            ],
        ),
        "programming-languages": (
            "Programming Languages",
            [
                "Statically typed languages check variable types at compile time, catching many errors before a program runs.",
                "Garbage collection automatically reclaims memory that a program can no longer reach.",
                "Functional programming emphasizes pure functions and immutable data over shared mutable state.",
                "A compiler translates source code into machine code or bytecode ahead of execution.",
                "Interpreted languages execute source code line by line without a separate compilation step.",
                "Concurrency primitives like goroutines or async/await let a program make progress on multiple tasks at once.",
                "Package managers resolve and install a project's third-party library dependencies.",
                "Type inference lets a compiler deduce a variable's type without an explicit annotation.",
                "A virtual machine like the JVM provides a portable runtime that abstracts away the underlying hardware.",
                "Language design trade-offs weigh performance, safety, and developer ergonomics against each other.",
                "Pattern matching lets code branch based on the shape of a value rather than a sequence of conditionals.",
                "Memory safety features prevent common bugs like buffer overflows and use-after-free errors.",
                "Generic programming lets a function or type operate over multiple concrete types without duplication.",
                "A linter analyzes source code statically to flag style issues and likely bugs before runtime.",
                "Closures capture variables from an enclosing scope so a function can use them after that scope has exited.",
                "Just-in-time compilation optimizes hot code paths while a program is already running.",
                "Language runtimes expose a foreign function interface to call code written in another language.",
                "Immutable data structures make concurrent programs easier to reason about by eliminating shared mutable state.",
            ],
        ),
    },
    "health": {
        "nutrition": (
            "Nutrition",
            [
                "A balanced diet provides carbohydrates, protein, and fat in proportions suited to an individual's energy needs.",
                "Dietary fiber slows digestion and helps regulate blood sugar and cholesterol levels.",
                "Micronutrients like vitamins and minerals are needed in small amounts but are essential for normal metabolism.",
                "Ultra-processed foods are often high in added sugar, sodium, and refined starch relative to whole foods.",
                "Protein is broken down into amino acids that the body uses to build and repair tissue.",
                "Hydration needs vary with body size, activity level, and ambient temperature.",
                "Omega-3 fatty acids found in fish and flaxseed are associated with cardiovascular benefits.",
                "Meal timing and portion size both influence how the body stores or burns consumed calories.",
                "A food label's serving size can differ substantially from how much a person actually eats in one sitting.",
                "Gut microbiota composition is shaped by long-term dietary patterns, especially fiber intake.",
                "Added sugars contribute calories without the vitamins and minerals found in whole fruit.",
                "Calcium and vitamin D work together to support bone density throughout life.",
                "Elimination diets remove suspected trigger foods temporarily to identify food sensitivities.",
                "Plant-based diets can meet protein needs by combining varied sources like legumes, grains, and nuts.",
                "Sodium intake above recommended levels is linked to elevated blood pressure in many people.",
                "Intermittent fasting restricts eating to specific windows rather than limiting specific food groups.",
                "Antioxidants in fruits and vegetables help neutralize reactive molecules produced by normal metabolism.",
                "Registered dietitians tailor nutrition plans to a person's health history, preferences, and goals.",
            ],
        ),
        "fitness": (
            "Fitness and Exercise",
            [
                "Progressive overload gradually increases training weight, volume, or intensity to keep driving adaptation.",
                "Cardiovascular exercise raises heart rate over a sustained period to improve aerobic capacity.",
                "Resistance training builds muscle by creating mechanical tension that stimulates tissue repair and growth.",
                "A proper warm-up increases blood flow and joint mobility before higher-intensity activity.",
                "Rest days allow muscle tissue to recover and adapt between training sessions.",
                "VO2 max measures the maximum rate at which the body can consume oxygen during intense exercise.",
                "High-intensity interval training alternates short bursts of maximal effort with brief recovery periods.",
                "Flexibility training through stretching helps maintain range of motion around a joint.",
                "Proper lifting form reduces the risk of injury by keeping joints in safe, stable positions.",
                "Resting heart rate tends to decrease as cardiovascular fitness improves over time.",
                "Periodization structures training into phases that vary intensity and volume across weeks or months.",
                "Delayed onset muscle soreness typically peaks one to two days after unfamiliar or intense exercise.",
                "Core strength supports posture and stabilizes the spine during everyday movement.",
                "Aerobic base training builds the capacity to sustain moderate effort for long durations.",
                "A fitness tracker estimates calories burned using heart rate and movement sensor data.",
                "Mobility drills target the range of motion needed for a specific sport or lifting movement.",
                "Balance training reduces fall risk by improving proprioception and stabilizing muscle activation.",
                "Consistency over weeks and months matters more for fitness outcomes than any single workout.",
            ],
        ),
        "mental-health": (
            "Mental Health",
            [
                "Cognitive behavioral therapy helps people identify and reframe unhelpful thought patterns.",
                "Chronic stress can elevate cortisol levels and affect sleep, mood, and immune function.",
                "Mindfulness practices train sustained, nonjudgmental attention to the present moment.",
                "Social support from friends and family is strongly associated with better mental health outcomes.",
                "Sleep deprivation impairs emotional regulation and next-day cognitive performance.",
                "Anxiety disorders involve persistent, excessive worry that interferes with daily functioning.",
                "A mental health professional can distinguish situational sadness from clinical depression.",
                "Journaling can help surface recurring thought patterns that contribute to distress.",
                "Physical activity has measurable positive effects on mood through multiple biological pathways.",
                "Burnout develops from prolonged, unmanaged workplace stress and is marked by exhaustion and cynicism.",
                "Therapeutic alliance, the trust between client and therapist, predicts treatment outcomes across modalities.",
                "Breathing exercises can activate the parasympathetic nervous system to reduce acute anxiety.",
                "Stigma around seeking mental health care remains a barrier for many people who could benefit from treatment.",
                "Grief follows no fixed timeline and can resurface around anniversaries or reminders.",
                "Screen time before bed can disrupt the natural release of melatonin and delay sleep onset.",
                "Peer support groups let people connect with others who share similar mental health experiences.",
                "Early childhood adversity is associated with elevated risk for certain mental health conditions in adulthood.",
                "Medication and psychotherapy are often combined for treating moderate to severe depression.",
            ],
        ),
        "medical-conditions": (
            "Medical Conditions",
            [
                "Type 2 diabetes develops when cells become resistant to insulin's effect on blood glucose.",
                "Hypertension is often called a silent condition because it frequently has no noticeable symptoms.",
                "Asthma causes airway inflammation that narrows the passages carrying air into the lungs.",
                "Autoimmune conditions occur when the immune system mistakenly attacks the body's own tissue.",
                "Atherosclerosis is the gradual buildup of plaque inside arterial walls over many years.",
                "Migraines are recurring headaches often accompanied by sensitivity to light and sound.",
                "Osteoarthritis results from the gradual breakdown of cartilage that cushions joints.",
                "Anemia occurs when the blood lacks enough healthy red blood cells to carry adequate oxygen.",
                "Chronic kidney disease is staged by how much filtering capacity the kidneys have lost.",
                "Seasonal allergies are triggered by an immune overreaction to airborne pollen or mold spores.",
                "Thyroid disorders can cause metabolism to run either abnormally fast or abnormally slow.",
                "Gastroesophageal reflux occurs when stomach acid flows backward into the esophagus.",
                "Sleep apnea repeatedly interrupts breathing during sleep, often without the person waking fully aware.",
                "Vaccines train the immune system to recognize a pathogen without causing the disease itself.",
                "Chronic inflammation is implicated in the progression of several common long-term diseases.",
                "Early screening can detect certain cancers before symptoms appear, improving treatment outcomes.",
                "Antibiotic resistance develops when bacteria evolve mechanisms to survive drugs that once killed them.",
                "A person's family medical history can inform which conditions they should be screened for earlier.",
            ],
        ),
    },
    "finance": {
        "investing": (
            "Investing",
            [
                "Diversification spreads investment risk across assets that do not all move together.",
                "Compound interest lets returns generate their own additional returns over time.",
                "Index funds passively track a market benchmark rather than trying to beat it through stock picking.",
                "Dollar-cost averaging invests a fixed amount at regular intervals regardless of price.",
                "A stock's price-to-earnings ratio compares its market price to its per-share earnings.",
                "Bonds pay periodic interest and return principal at maturity, generally with lower risk than stocks.",
                "Asset allocation divides a portfolio among stocks, bonds, and other classes based on risk tolerance and timeline.",
                "Dividend-paying stocks distribute a portion of company profits directly to shareholders.",
                "Market volatility measures how much an asset's price fluctuates over a given period.",
                "A brokerage account is the vehicle through which an individual buys and sells securities.",
                "Rebalancing periodically restores a portfolio's target mix of assets after markets drift it out of line.",
                "Risk tolerance reflects how much portfolio volatility an investor can accept without abandoning their plan.",
                "Exchange-traded funds trade throughout the day like stocks while holding a basket of underlying assets.",
                "A bear market is commonly defined as a broad decline of twenty percent or more from recent highs.",
                "Expense ratios are the annual fees a fund charges, expressed as a percentage of assets under management.",
                "Tax-advantaged retirement accounts let investment growth compound without annual tax drag.",
                "Value investing seeks companies trading below an estimate of their intrinsic worth.",
                "A time horizon of decades allows an investor to ride out short-term market downturns more comfortably.",
            ],
        ),
        "personal-finance": (
            "Personal Finance",
            [
                "An emergency fund covers unexpected expenses without forcing high-interest borrowing.",
                "A budget tracks income against expenses to keep spending aligned with financial goals.",
                "Credit scores summarize borrowing history and influence the interest rates a lender will offer.",
                "Compound interest on debt can make minimum credit card payments extremely costly over time.",
                "Automating savings transfers removes the temptation to skip contributions in a given month.",
                "A debt payoff strategy like the avalanche method targets the highest-interest balance first.",
                "Net worth is calculated as total assets minus total liabilities at a point in time.",
                "Insurance transfers the financial risk of rare but costly events to a pooled group of policyholders.",
                "A fixed-rate mortgage locks in the same interest rate for the entire loan term.",
                "Reviewing recurring subscriptions periodically can surface spending that no longer matches a person's needs.",
                "An emergency fund is typically sized to cover three to six months of essential expenses.",
                "Employer retirement matching is effectively additional compensation tied to an employee's own contributions.",
                "A debt-to-income ratio compares monthly debt payments to gross monthly income.",
                "Sinking funds set aside money in advance for predictable future expenses like car repairs.",
                "Refinancing a loan can lower the interest rate but may reset the amortization schedule.",
                "Financial literacy education covers budgeting, credit, saving, and investing fundamentals.",
                "A will specifies how a person's assets should be distributed after their death.",
                "Lifestyle inflation is the tendency to increase spending as income rises, often eroding savings potential.",
            ],
        ),
        "markets": (
            "Financial Markets",
            [
                "Central banks influence borrowing costs economy-wide by setting a benchmark short-term interest rate.",
                "Inflation measures how quickly the average price level for goods and services rises over time.",
                "A yield curve plots interest rates for bonds of different maturities at a single point in time.",
                "Market liquidity describes how easily an asset can be bought or sold without moving its price.",
                "Gross domestic product estimates the total value of goods and services produced within a country.",
                "Currency exchange rates fluctuate based on relative interest rates, trade flows, and investor sentiment.",
                "A recession is commonly associated with a sustained decline in economic activity across many sectors.",
                "Commodity prices for goods like oil and wheat respond to shifts in global supply and demand.",
                "Market indices aggregate the performance of a basket of securities into a single benchmark number.",
                "Unemployment figures are a lagging indicator that often continues rising after a recession has ended.",
                "Quantitative easing involves a central bank purchasing assets to inject liquidity into the financial system.",
                "Corporate earnings reports reveal how a company's revenue and profit changed over the prior quarter.",
                "Credit spreads widen when investors demand more compensation for holding riskier corporate debt.",
                "Stock market circuit breakers pause trading temporarily after sharp, rapid price declines.",
                "Consumer sentiment surveys gauge how optimistic households feel about their financial future.",
                "Trade deficits occur when a country imports more in value than it exports over a period.",
                "Market makers provide liquidity by continuously quoting both buy and sell prices for a security.",
                "Sovereign credit ratings assess a government's ability and willingness to repay its debt obligations.",
            ],
        ),
    },
    "sports": {
        "basketball": (
            "Basketball",
            [
                "A basketball possession ends when a team scores, turns the ball over, or the shot clock expires.",
                "Pick-and-roll actions create scoring opportunities by forcing defenders to choose between two threats.",
                "Advanced metrics like true shooting percentage account for the extra value of three-point shots and free throws.",
                "Defensive rotations require players to communicate quickly when a teammate is beaten off the dribble.",
                "A team's offensive rating measures points scored per hundred possessions.",
                "Rebounding position is won largely through anticipation and body positioning rather than height alone.",
                "The pace of a game refers to the number of possessions each team uses per forty-eight minutes.",
                "Zone defenses assign players to guard areas of the court rather than specific opposing players.",
                "Fast breaks exploit a temporary numbers advantage before the defense can fully set up.",
                "Free-throw shooting is practiced extensively because it directly converts fouls into guaranteed points.",
                "Screen assists occur when a player's off-ball screen directly leads to a teammate's basket.",
                "A team's net rating is the difference between its offensive and defensive ratings.",
                "Help defense requires a player to leave their assignment briefly to stop a more dangerous scoring threat.",
                "Player efficiency ratings combine box score statistics into a single per-minute production estimate.",
                "Isolation plays clear space for one offensive player to attack a single defender one-on-one.",
                "Transition defense focuses on getting back quickly before the opposing team can attack in the open floor.",
                "Shot selection analysis has pushed many teams toward taking more three-pointers and shots at the rim.",
                "Timeouts allow a coaching staff to draw up a specific play for a critical late-game possession.",
            ],
        ),
        "soccer": (
            "Soccer",
            [
                "A soccer formation describes how a team's players are arranged across defense, midfield, and attack.",
                "Possession-based tactics prioritize retaining the ball to control the tempo of a match.",
                "The offside rule prevents an attacker from being in an advanced position when the ball is played to them.",
                "Pressing high up the pitch aims to win the ball back quickly in the opponent's defensive third.",
                "Set pieces like free kicks and corners create scoring chances from a stopped, restarted play.",
                "A false nine drops deeper than a traditional striker to link attacking play through midfield.",
                "Expected goals estimates the quality of a scoring chance based on shot location and situation.",
                "Full-backs increasingly join the attack by overlapping with wingers on either flank.",
                "A counter-attack exploits open space left behind when the opposing team commits players forward.",
                "Man-marking assigns a defender to track one specific opposing player throughout the match.",
                "Extra time and penalty shootouts resolve knockout matches that finish level after regulation.",
                "A team's defensive line height affects how much space is left behind for attackers to exploit.",
                "Wing play stretches the opposing defense wide to open central passing lanes.",
                "Substitutions let a manager change tactics or manage player fatigue during a match.",
                "Ball retention under pressure relies on quick, accurate short passing between closely spaced teammates.",
                "A club's youth academy develops players internally rather than relying solely on transfers.",
                "Video assistant referees review contentious decisions like penalties and offside calls.",
                "Transfer windows are fixed periods during which clubs can buy and sell player contracts.",
            ],
        ),
        "running": (
            "Running and Endurance",
            [
                "Marathon training typically builds weekly mileage gradually to reduce the risk of overuse injury.",
                "Tempo runs are sustained efforts just below race pace intended to improve lactate threshold.",
                "Negative splitting means running the second half of a race faster than the first half.",
                "Running economy describes how efficiently a runner uses oxygen at a given pace.",
                "A taper reduces training volume in the final weeks before a race to arrive fresh on race day.",
                "Interval training alternates fast repeats with recovery jogs to improve speed and aerobic capacity.",
                "Carbohydrate loading increases glycogen stores in the days before a long endurance event.",
                "Proper running form minimizes wasted vertical motion and excessive braking forces on foot strike.",
                "Heart rate zones divide training intensity into ranges tied to specific physiological adaptations.",
                "Long slow runs build the aerobic base that supports faster, more intense training later.",
                "Hydration and electrolyte strategy during a race depend on distance, pace, and ambient temperature.",
                "Recovery runs are performed at an easy pace to promote blood flow without adding significant stress.",
                "A running gait analysis can identify mechanical inefficiencies that contribute to recurring injuries.",
                "Altitude training increases red blood cell production, which can boost oxygen-carrying capacity at sea level.",
                "Pacing strategy in a race balances the risk of starting too fast against leaving performance unused.",
                "Cross-training with cycling or swimming maintains aerobic fitness while reducing impact-related stress.",
                "The wall in a marathon refers to a sudden loss of energy typically tied to depleted glycogen stores.",
                "Shoe cushioning and drop influence how impact forces are distributed through the foot and leg.",
            ],
        ),
    },
    "science": {
        "astronomy": (
            "Astronomy and Space",
            [
                "A light-year measures the distance light travels through a vacuum in one year.",
                "Stars generate energy through nuclear fusion, converting hydrogen into helium in their cores.",
                "A galaxy is a gravitationally bound collection of stars, gas, dust, and dark matter.",
                "Exoplanets are detected indirectly through techniques like the transit method and radial velocity.",
                "A black hole's event horizon marks the boundary beyond which nothing, including light, can escape.",
                "The cosmic microwave background is faint radiation left over from the early universe.",
                "Space telescopes avoid atmospheric distortion, allowing sharper images than most ground-based observatories.",
                "A planet's orbit is shaped by the gravitational pull of the star it revolves around.",
                "Supernovae occur when a massive star exhausts its fuel and collapses catastrophically.",
                "The habitable zone around a star is the range of orbital distances where liquid water could exist.",
                "Spectroscopy analyzes light from distant objects to determine their composition and motion.",
                "Dark matter does not emit light but its gravitational effects can be inferred from galaxy rotation.",
                "A comet's tail forms when solar radiation vaporizes ice as the comet nears the sun.",
                "Redshift indicates that a distant galaxy is moving away from an observer, stretching its light's wavelength.",
                "Planetary rings are composed of countless small particles of ice and rock in orbit.",
                "Solar flares release bursts of radiation that can disrupt satellite communication and power grids on Earth.",
                "A meteor shower occurs when Earth passes through debris left behind by a comet's orbit.",
                "The Kuiper Belt is a region beyond Neptune containing icy bodies including several dwarf planets.",
            ],
        ),
        "biology": (
            "Biology",
            [
                "DNA encodes genetic instructions using sequences of four nucleotide bases.",
                "Natural selection favors traits that improve an organism's likelihood of survival and reproduction.",
                "Cells use mitochondria to convert nutrients into usable chemical energy through respiration.",
                "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
                "Enzymes speed up biochemical reactions by lowering the energy required for them to proceed.",
                "An ecosystem's food web describes the flow of energy between producers, consumers, and decomposers.",
                "Genetic mutations introduce variation that natural selection can act upon over generations.",
                "Cellular respiration breaks down glucose in the presence of oxygen to release stored energy.",
                "Homeostasis refers to an organism's ability to maintain stable internal conditions despite external change.",
                "Symbiotic relationships between species can be mutually beneficial, parasitic, or commensal.",
                "Proteins fold into specific three-dimensional shapes that determine their biological function.",
                "Biodiversity within an ecosystem tends to increase its resilience to environmental disturbance.",
                "Stem cells can differentiate into multiple specialized cell types during development.",
                "Species evolve adaptations over many generations in response to selective pressure from their environment.",
                "The immune system distinguishes the body's own cells from foreign pathogens using surface markers.",
                "Population dynamics are shaped by birth rates, death rates, and migration between habitats.",
                "Gene expression is regulated by factors that determine which genes are turned on in a given cell.",
                "Coral reefs support disproportionately high biodiversity relative to the area of ocean they occupy.",
            ],
        ),
        "climate": (
            "Climate Science",
            [
                "Greenhouse gases trap outgoing infrared radiation, warming the lower atmosphere over time.",
                "Ocean currents redistribute heat around the planet, moderating regional climates far from the equator.",
                "Ice cores preserve trapped air bubbles that record atmospheric composition from thousands of years ago.",
                "Rising global temperatures are associated with more frequent and intense extreme weather events.",
                "Carbon sinks like forests and oceans absorb more carbon dioxide than they release.",
                "Sea level rise results from both thermal expansion of seawater and melting land ice.",
                "Climate models simulate how the atmosphere, oceans, and ice interact over long time scales.",
                "Deforestation reduces a landscape's capacity to absorb atmospheric carbon dioxide.",
                "The greenhouse effect is a natural process that becomes a concern when gas concentrations rise sharply.",
                "Ocean acidification occurs as seawater absorbs excess atmospheric carbon dioxide, lowering its pH.",
                "Weather describes short-term atmospheric conditions, while climate describes long-term average patterns.",
                "Renewable energy sources like solar and wind produce electricity without direct carbon dioxide emissions.",
                "Permafrost thaw can release previously trapped methane, a potent greenhouse gas.",
                "Climate feedback loops can either amplify or dampen an initial warming or cooling trend.",
                "Extreme heat events pose disproportionate health risks to elderly and outdoor-working populations.",
                "Glacial retreat is tracked over decades using satellite imagery and on-the-ground measurements.",
                "Carbon pricing mechanisms attach an economic cost to greenhouse gas emissions to influence behavior.",
                "Coastal communities face compounding risks from sea level rise and more intense storm surges.",
            ],
        ),
    },
    "cooking": {
        "baking": (
            "Baking",
            [
                "Gluten development in dough depends on the amount of mixing and the protein content of the flour.",
                "Leavening agents like yeast and baking soda produce gas that makes baked goods rise.",
                "Oven temperature accuracy strongly affects how bread crust color and crumb texture develop.",
                "Creaming butter and sugar together incorporates air that helps a cake rise and stay tender.",
                "Resting dough allows gluten strands to relax, making it easier to shape without tearing.",
                "Proofing gives yeast time to ferment sugars and produce carbon dioxide before baking.",
                "The Maillard reaction between sugars and proteins produces browning and complex flavor at high heat.",
                "Precise measurement matters more in baking than in most cooking because ratios drive chemical reactions.",
                "Cold butter creates flaky layers in pastry by melting during baking and leaving steam pockets.",
                "Autolyse rests flour and water together before mixing to improve dough extensibility.",
                "Egg proteins coagulate with heat, providing structure in custards, cakes, and meringues.",
                "A high hydration dough produces an open, irregular crumb typical of rustic artisan bread.",
                "Sifting dry ingredients removes lumps and evenly distributes leavening agents throughout a batter.",
                "Sugar affects more than sweetness in baking; it also influences moisture retention and browning.",
                "Steam injection in the first minutes of baking helps bread develop a crisp, glossy crust.",
                "Overmixing batter can overdevelop gluten, producing a tough rather than tender crumb.",
                "Salt strengthens gluten structure and also slows yeast fermentation slightly.",
                "Blind baking a pie crust prevents it from becoming soggy under a wet filling.",
            ],
        ),
        "cuisine": (
            "World Cuisine",
            [
                "Umami, often described as savory depth, is one of the five basic tastes alongside sweet, sour, salty, and bitter.",
                "Fermentation is used across many cuisines to preserve food and develop complex flavors.",
                "Regional cuisines are shaped by locally available ingredients, climate, and historical trade routes.",
                "A sofrito or mirepoix base of aromatics forms the flavor foundation for many braises and sauces.",
                "Street food traditions often reflect a city's cultural history and immigrant communities.",
                "Spice blends vary widely by region, from garam masala in South Asia to za'atar in the Levant.",
                "Fermented condiments like soy sauce and fish sauce add concentrated savory flavor in small amounts.",
                "Communal dining styles, where dishes are shared, are common across many East and Southeast Asian cuisines.",
                "Braising combines dry searing with slow, moist cooking to tenderize tougher cuts of meat.",
                "Rice is a staple starch across many cuisines and can be prepared dozens of distinct ways.",
                "Pickling uses acid or salt brine to preserve vegetables while adding tang and crunch.",
                "A cuisine's use of chili heat ranges enormously, from mild to intensely spicy across different regions.",
                "Street markets often showcase a region's freshest seasonal produce alongside prepared foods.",
                "Broths and stocks simmered for hours form the base of many soups across global culinary traditions.",
                "Grilling over an open flame imparts smoky char that is central to many barbecue traditions worldwide.",
                "Dumplings appear in many independent culinary traditions, from pierogi to gyoza to empanadas.",
                "Balancing sweet, sour, salty, and spicy elements is a defining feature of many Southeast Asian dishes.",
                "Food preservation techniques like curing and smoking predate refrigeration by centuries.",
            ],
        ),
        "knife-skills": (
            "Knife Skills and Prep",
            [
                "A sharp knife is safer than a dull one because it requires less force and slips less predictably.",
                "The claw grip tucks fingertips under while holding food, protecting them from the blade.",
                "A julienne cut produces thin, matchstick-shaped pieces of vegetable.",
                "Honing a blade on a steel realigns the edge between full sharpening sessions.",
                "A chef's knife is versatile enough to handle most chopping, slicing, and mincing tasks.",
                "Mise en place means prepping and organizing all ingredients before cooking actually begins.",
                "A rocking motion with the knife tip anchored on the board speeds up fine mincing.",
                "Cutting board stability, often aided by a damp towel underneath, reduces the risk of slipping.",
                "A brunoise cut is a very fine dice, typically produced by first julienning and then cross-cutting.",
                "Bias cuts slice vegetables at an angle to increase surface area for faster, more even cooking.",
                "Whetstone sharpening removes small amounts of metal to restore a blade's fine edge.",
                "A paring knife's small blade is suited to detailed work like peeling and trimming.",
                "Consistent cut sizes help ingredients cook evenly in the same amount of time.",
                "A rocking chop technique keeps the knife tip in contact with the board throughout the cut.",
                "Onions are typically cut root-to-tip first to keep layers intact before cross-cutting into dice.",
                "A serrated knife's teeth saw through crusty bread without crushing the softer interior.",
                "Proper knife grip places the thumb and forefinger on the blade itself, not just the handle.",
                "Boning knives have thin, flexible blades designed to separate meat cleanly from bone.",
            ],
        ),
    },
}


def build_documents() -> list[Document]:
    rng = random.Random(RANDOM_SEED)
    docs: list[Document] = []
    for domain, subtopics in FACT_BANKS.items():
        for subtopic, (title, facts) in subtopics.items():
            n_facts = len(facts)
            for i in range(DOCS_PER_SUBTOPIC):
                # Rotate the starting point and sample a contiguous-ish window with
                # light shuffling, so each doc in a subtopic has real lexical
                # variety while staying topically coherent.
                k = rng.randint(5, min(8, n_facts))
                start = (i * 3) % n_facts
                ordered = facts[start:] + facts[:start]
                rng.shuffle(ordered)
                chosen = ordered[:k]
                text = " ".join(chosen)
                doc_id = f"{domain}-{subtopic}-{i:03d}"
                docs.append(
                    Document(
                        id=doc_id,
                        domain=domain,
                        subtopic=subtopic,
                        title=f"{title} Notes #{i + 1}",
                        text=text,
                    )
                )
    rng.shuffle(docs)
    return docs


def main() -> None:
    docs = build_documents()
    write_corpus(docs, CORPUS_PATH)
    print(f"Wrote {len(docs)} documents to {CORPUS_PATH}")
    domains = sorted({d.domain for d in docs})
    subtopics = sorted({(d.domain, d.subtopic) for d in docs})
    print(f"Domains: {domains}")
    print(f"Subtopics: {len(subtopics)}")


if __name__ == "__main__":
    main()
