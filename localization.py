"""
Display-only Arabic localization for GreenLens.

IMPORTANT: nothing in this module is ever passed back to the model, the
similarity search or the recommendation engine. It only converts the
English text those components already return into Arabic *for display*.
Original class identifiers (e.g. "Tomato_Late_Blight") are never modified.

If a string has no Arabic translation, the original English text is
returned unchanged, so a missing entry can never break the UI.
"""

import re

# ---------------------------------
# Plant names
# ---------------------------------
PLANT_AR = {
    "Apple": "التفاح",
    "Corn": "الذرة",
    "Tomato": "الطماطم",
    "Unknown": "غير معروف",
}

# ---------------------------------
# Condition (disease) names
# ---------------------------------
CONDITION_AR = {
    "Apple Scab": "جرب التفاح",
    "Black Rot": "العفن الأسود",
    "Cedar Apple Rust": "صدأ التفاح والأرز",
    "Cercospora Leaf Spot": "تبقع الأوراق السركوسبوري",
    "Common Rust": "الصدأ الشائع",
    "Northern Leaf Blight": "لفحة الأوراق الشمالية",
    "Bacterial Spot": "التبقع البكتيري",
    "Early Blight": "اللفحة المبكرة",
    "Late Blight": "اللفحة المتأخرة",
    "Septoria Leaf Spot": "تبقع أوراق السبتوريا",
    "Possible Tomato Yellow Leaf Curl Virus": "احتمال الإصابة بفيروس تجعد واصفرار أوراق الطماطم",
    "No disease detected by the model": "لم يرصد النموذج أي مرض",
    "Uncertain prediction": "تشخيص غير مؤكد",
    "Unrecognized class": "فئة غير معروفة",
}

# ---------------------------------
# Status lines
# ---------------------------------
STATUS_AR = {
    "Possible disease": "مرض محتمل",
    "Model classified as healthy": "صنّف النموذج النبات على أنه سليم",
    "Possible disease requiring prompt assessment": "مرض محتمل يستدعي تقييمًا سريعًا",
    "Possible disease requiring confirmation": "مرض محتمل يحتاج إلى تأكيد",
    "Low model confidence": "ثقة النموذج منخفضة",
    "No matching recommendation found": "لم يُعثر على توصية مطابقة",
}

# ---------------------------------
# Care recommendation sentences (every string found in recommendations.py)
# ---------------------------------
ADVICE_AR = {
    # Apple Scab
    "Inspect leaves and fruits for dark or olive-colored lesions.": "افحص الأوراق والثمار بحثًا عن بقع داكنة أو بلون الزيتون.",
    "Remove fallen infected leaves where practical.": "أزل الأوراق المصابة المتساقطة متى أمكن ذلك.",
    "Improve airflow around the tree.": "حسّن حركة الهواء حول الشجرة.",
    "Consult an agricultural specialist if symptoms spread.": "استشر مختصًا زراعيًا إذا انتشرت الأعراض.",
    # Apple Black Rot
    "Inspect leaves, branches, and fruits for dark lesions.": "افحص الأوراق والأغصان والثمار بحثًا عن بقع داكنة.",
    "Remove affected plant material where appropriate.": "أزل الأجزاء النباتية المصابة عند الحاجة.",
    "Monitor the tree for spreading symptoms.": "راقب الشجرة لرصد أي انتشار للأعراض.",
    "Seek local agricultural advice if symptoms worsen.": "اطلب مشورة زراعية محلية إذا ازدادت الأعراض سوءًا.",
    # Cedar Apple Rust
    "Inspect leaves for orange or yellow spots.": "افحص الأوراق بحثًا عن بقع برتقالية أو صفراء.",
    "Check other leaves for similar symptoms.": "تحقق من ظهور أعراض مشابهة على الأوراق الأخرى.",
    "Monitor symptom progression.": "تابع تطور الأعراض.",
    "Consult an agricultural specialist for confirmation.": "استشر مختصًا زراعيًا للتأكد من التشخيص.",
    # Apple Healthy
    "Continue regular plant inspection.": "واصل فحص النبات بانتظام.",
    "Maintain suitable watering and growing conditions.": "حافظ على ريّ وظروف نمو مناسبة.",
    "Monitor new leaves and fruit for changes.": "راقب الأوراق والثمار الجديدة لرصد أي تغيّرات.",
    # Corn Cercospora
    "Inspect leaves for elongated lesions.": "افحص الأوراق بحثًا عن بقع مستطيلة.",
    "Monitor other plants for similar symptoms.": "راقب النباتات الأخرى لرصد أعراض مشابهة.",
    "Reduce prolonged leaf wetness where practical.": "قلّل من بقاء الأوراق مبللة لفترات طويلة متى أمكن ذلك.",
    "Seek agricultural advice if symptoms spread.": "اطلب مشورة زراعية إذا انتشرت الأعراض.",
    # Corn Common Rust
    "Inspect leaves for rust-colored pustules.": "افحص الأوراق بحثًا عن بثرات بلون الصدأ.",
    "Check multiple leaves for symptoms.": "افحص عدة أوراق للتحقق من وجود الأعراض.",
    "Monitor the crop for progression.": "تابع المحصول لرصد تطور الإصابة.",
    "Consult an agricultural specialist if symptoms are widespread.": "استشر مختصًا زراعيًا إذا كانت الأعراض واسعة الانتشار.",
    # Corn Healthy
    "Monitor crop growth and leaf condition.": "راقب نمو المحصول وحالة الأوراق.",
    "Maintain suitable irrigation.": "حافظ على ريّ مناسب.",
    "Inspect plants regularly for new symptoms.": "افحص النباتات بانتظام بحثًا عن أعراض جديدة.",
    # Corn Northern Leaf Blight
    "Inspect leaves for long, elongated lesions.": "افحص الأوراق بحثًا عن بقع طويلة ممتدة.",
    "Check nearby plants for similar symptoms.": "تحقق من ظهور أعراض مشابهة على النباتات المجاورة.",
    "Monitor changes in affected leaves.": "راقب التغيّرات في الأوراق المصابة.",
    "Seek agricultural advice about management options.": "اطلب مشورة زراعية بشأن خيارات المعالجة.",
    # Tomato Bacterial Spot
    "Inspect leaves and fruits for small dark spots.": "افحص الأوراق والثمار بحثًا عن بقع صغيرة داكنة.",
    "Avoid handling wet plants where practical.": "تجنّب لمس النباتات وهي مبللة متى أمكن ذلك.",
    "Clean gardening tools before using them on other plants.": "نظّف أدوات البستنة قبل استخدامها على نباتات أخرى.",
    "Seek expert confirmation before choosing a treatment.": "احصل على تأكيد من مختص قبل اختيار العلاج.",
    # Tomato Early Blight
    "Inspect lower leaves for dark spots with concentric rings.": "افحص الأوراق السفلية بحثًا عن بقع داكنة ذات حلقات متحدة المركز.",
    "Remove severely affected leaves when appropriate.": "أزل الأوراق شديدة الإصابة عند الحاجة.",
    "Avoid overhead watering where practical.": "تجنّب الري من الأعلى متى أمكن ذلك.",
    "Monitor whether symptoms spread.": "راقب ما إذا كانت الأعراض تنتشر.",
    # Tomato Healthy
    "Inspect leaves, stems, and fruits regularly.": "افحص الأوراق والسيقان والثمار بانتظام.",
    "Maintain consistent, suitable watering.": "حافظ على ريّ منتظم ومناسب.",
    "Provide adequate sunlight and airflow.": "وفّر قدرًا كافيًا من ضوء الشمس والتهوية.",
    "Monitor new growth for changes.": "راقب النمو الجديد لرصد أي تغيّرات.",
    # Tomato Late Blight
    "Inspect leaves and stems for expanding dark lesions.": "افحص الأوراق والسيقان بحثًا عن بقع داكنة آخذة في الاتساع.",
    "Avoid moving potentially infected plant material.": "تجنّب نقل المواد النباتية التي قد تكون مصابة.",
    "Seek prompt agricultural advice because this disease can spread rapidly.": "اطلب مشورة زراعية عاجلة لأن هذا المرض قد ينتشر بسرعة.",
    "Do not rely on the image prediction alone to choose treatment.": "لا تعتمد على التشخيص المبني على الصورة وحده لاختيار العلاج.",
    # Tomato Septoria
    "Inspect leaves for small circular spots with pale centers.": "افحص الأوراق بحثًا عن بقع دائرية صغيرة ذات مراكز باهتة.",
    "Remove affected lower leaves when appropriate.": "أزل الأوراق السفلية المصابة عند الحاجة.",
    "Monitor for additional spots.": "راقب ظهور بقع إضافية.",
    # Tomato Yellow Leaf Curl Virus
    "Inspect for leaf curling, yellowing, and reduced growth.": "افحص النبات بحثًا عن تجعّد الأوراق واصفرارها وضعف النمو.",
    "Monitor for whiteflies, which can transmit this virus.": "راقب وجود الذباب الأبيض الذي قد ينقل هذا الفيروس.",
    "Seek agricultural advice to confirm the cause.": "اطلب مشورة زراعية لتأكيد السبب.",
    # Fallback: low confidence
    "Take a clear photo in good lighting.": "التقط صورة واضحة في إضاءة جيدة.",
    "Focus on the affected leaf and avoid blurry images.": "ركّز على الورقة المصابة وتجنّب الصور الضبابية.",
    "Try another image showing the symptoms clearly.": "جرّب صورة أخرى تُظهر الأعراض بوضوح.",
    "Consult an agricultural specialist if symptoms persist.": "استشر مختصًا زراعيًا إذا استمرت الأعراض.",
    # Fallback: unrecognized class
    "Check the predicted class name.": "تحقق من اسم الفئة المتوقعة.",
    "Upload a clear image if necessary.": "ارفع صورة واضحة عند الحاجة.",
}

# ---------------------------------
# Disclaimers
# ---------------------------------
DISCLAIMER_AR = {
    "AI-generated suggestions are not a confirmed diagnosis. "
    "Model confidence is not necessarily a calibrated probability.":
        "الاقتراحات المُولَّدة بالذكاء الاصطناعي ليست تشخيصًا مؤكدًا، "
        "كما أن درجة ثقة النموذج ليست بالضرورة احتمالًا دقيقًا.",
    "The prediction is uncertain. These suggestions are not a confirmed diagnosis.":
        "التشخيص غير مؤكد، وهذه الاقتراحات ليست تشخيصًا مؤكدًا.",
    "This class is not available in the recommendation database.":
        "هذه الفئة غير متوفرة في قاعدة بيانات التوصيات.",
}

# ---------------------------------
# Model class names (display only)
# Keys are normalised labels: underscores -> spaces, whitespace collapsed.
# ---------------------------------
CLASS_AR = {
    "Apple Apple Scab": "التفاح - جرب التفاح",
    "Apple Black Rot": "التفاح - العفن الأسود",
    "Apple Cedar Apple Rust": "التفاح - صدأ التفاح والأرز",
    "Apple Healthy": "التفاح - سليم",
    "Corn (maize) Cercospora Leaf Spot": "الذرة - تبقع الأوراق السركوسبوري",
    "Corn (maize) Common Rust": "الذرة - الصدأ الشائع",
    "Corn (maize) Healthy": "الذرة - سليمة",
    "Corn (maize) Northern Leaf Blight": "الذرة - لفحة الأوراق الشمالية",
    "Tomato Bacterial Spot": "الطماطم - التبقع البكتيري",
    "Tomato Early Blight": "الطماطم - اللفحة المبكرة",
    "Tomato Healthy": "الطماطم - سليمة",
    "Tomato Late Blight": "الطماطم - اللفحة المتأخرة",
    "Tomato Septoria Leaf Spot": "الطماطم - تبقع أوراق السبتوريا",
    "Tomato Yellow Leaf Curl Virus": "الطماطم - فيروس تجعد واصفرار الأوراق",
}


def normalize_class_label(name):
    """'Corn_(maize)_Common_Rust_' -> 'Corn (maize) Common Rust'.

    The original code used a plain replace('_', ' ') which left a trailing
    space for class names ending in '_' and silently broke the translation
    lookup for that class.
    """
    return re.sub(r"\s+", " ", str(name).replace("_", " ")).strip()


def class_label(name, arabic=False):
    """Human-readable class name for display; the identifier is untouched."""
    label = normalize_class_label(name)
    return CLASS_AR.get(label, label) if arabic else label


def _lookup(table, text):
    return table.get(text, text)


def localize_recommendation(recommendation, arabic=False):
    """Return a *copy* of a recommendation dict with display text translated.

    The input dict is never modified. Unknown strings stay in English.
    """
    if not arabic:
        return recommendation

    localized = dict(recommendation)
    localized["plant"] = _lookup(PLANT_AR, recommendation.get("plant", "Unknown"))
    localized["condition"] = _lookup(CONDITION_AR, recommendation.get("condition", "Unknown"))
    if recommendation.get("status"):
        localized["status"] = _lookup(STATUS_AR, recommendation["status"])
    localized["recommendations"] = [
        _lookup(ADVICE_AR, item) for item in recommendation.get("recommendations", [])
    ]
    if recommendation.get("disclaimer"):
        localized["disclaimer"] = _lookup(DISCLAIMER_AR, recommendation["disclaimer"])
    return localized
