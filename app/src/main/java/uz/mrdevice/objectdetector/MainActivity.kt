package uz.mrdevice.objectdetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.util.Size
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.Lifecycle
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

class MainActivity : ComponentActivity() {
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val modelExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    @Volatile private var detector: ObjectDetector? = null
    @Volatile private var screenIsActive = false

    private lateinit var previewView: PreviewView
    private lateinit var overlay: DetectionOverlay
    private lateinit var status: TextView
    private lateinit var count: TextView

    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var cameraStarted = false
    @Volatile private var cameraStarting = false
    @Volatile private var cameraGeneration = 0
    private val objectAgentEngine = ObjectAgentEngine()

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            status.text = "Kamera ruxsati berilmadi. Sozlamalardan kameraga ruxsat bering."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildInterface()
        screenIsActive = true
        loadDetectionModel()
        requestCameraAccessAndStart()
    }

    private fun buildInterface() {
        WindowCompat.setDecorFitsSystemWindows(window, false)

        val page = Color.rgb(8, 13, 24)
        val surface = Color.rgb(17, 25, 41)
        val surfaceRaised = Color.rgb(23, 34, 53)
        val accent = Color.rgb(0, 220, 170)
        val muted = Color.rgb(151, 166, 188)
        val primary = Color.rgb(239, 245, 252)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(12))
            setBackgroundColor(page)
            clipToPadding = false
        }
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(
                dp(16) + systemBars.left,
                dp(12) + systemBars.top,
                dp(16) + systemBars.right,
                dp(12) + systemBars.bottom
            )
            insets
        }

        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(2), 0, dp(12))
        }
        val logo = TextView(this).apply {
            setCompoundDrawablesWithIntrinsicBounds(R.drawable.bilmasvoy_logo, 0, 0, 0)
            compoundDrawablePadding = dp(10)
            text = "BILMASVOY"
            textSize = 20f
            setTextColor(primary)
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            gravity = Gravity.CENTER_VERTICAL
        }
        val version = TextView(this).apply {
            text = "AI VISION"
            textSize = 10f
            letterSpacing = 0.12f
            setTextColor(accent)
            setPadding(dp(10), dp(7), dp(10), dp(7))
            background = roundedBackground(surfaceRaised, 18f, accent, dp(1))
        }
        header.addView(logo, LinearLayout.LayoutParams(0, dp(42), 1f))
        header.addView(version)

        val statusCard = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(11), dp(14), dp(11))
            background = roundedBackground(surface, 16f)
        }
        val statusDot = View(this).apply {
            background = roundedBackground(accent, 20f)
        }
        statusCard.addView(statusDot, LinearLayout.LayoutParams(dp(8), dp(8)))
        status = TextView(this).apply {
            text = "Sun’iy intellekt modeli yuklanmoqda…"
            textSize = 13f
            setTextColor(primary)
            setPadding(dp(10), 0, 0, 0)
            gravity = Gravity.CENTER_VERTICAL
        }
        statusCard.addView(status, LinearLayout.LayoutParams(0, -2, 1f))

        val cameraTitle = TextView(this).apply {
            text = "JONLI KUZATUV"
            textSize = 11f
            letterSpacing = 0.12f
            setTextColor(muted)
            setPadding(dp(2), dp(14), dp(2), dp(7))
        }

        val cameraCard = FrameLayout(this).apply {
            background = roundedBackground(surface, 20f)
            setPadding(dp(5), dp(5), dp(5), dp(5))
            clipToOutline = true
        }
        val cameraFrame = FrameLayout(this).apply {
            background = roundedBackground(Color.BLACK, 15f)
            clipToOutline = true
        }
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
            contentDescription = "Jonli kamera tasviri"
        }
        overlay = DetectionOverlay(this)
        cameraFrame.addView(previewView, FrameLayout.LayoutParams(-1, -1))
        cameraFrame.addView(overlay, FrameLayout.LayoutParams(-1, -1))
        val liveLabel = TextView(this).apply {
            text = "●  LIVE"
            textSize = 10f
            letterSpacing = 0.08f
            setTextColor(Color.WHITE)
            setPadding(dp(9), dp(6), dp(9), dp(6))
            background = roundedBackground(Color.argb(185, 8, 18, 30), 18f)
        }
        cameraFrame.addView(liveLabel, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.END).apply {
            setMargins(0, dp(10), dp(10), 0)
        })
        cameraCard.addView(cameraFrame, FrameLayout.LayoutParams(-1, -1))

        val statsTitle = TextView(this).apply {
            text = "ANIQLASH STATISTIKASI"
            textSize = 11f
            letterSpacing = 0.12f
            setTextColor(muted)
            setPadding(dp(2), dp(13), dp(2), dp(7))
        }
        val statsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val objectTile = createStatTile("TOPILDI", "0 ta", accent, surface)
        count = objectTile.second
        val modelTile = createStatTile("AI MODEL", "Lite2", primary, surface)
        val classTile = createStatTile("SINFLAR", "80", primary, surface)
        statsRow.addView(objectTile.first, LinearLayout.LayoutParams(0, dp(72), 1f).apply {
            marginEnd = dp(7)
        })
        statsRow.addView(modelTile.first, LinearLayout.LayoutParams(0, dp(72), 1f).apply {
            marginEnd = dp(7)
        })
        statsRow.addView(classTile.first, LinearLayout.LayoutParams(0, dp(72), 1f))

        val footer = TextView(this).apply {
            text = "Buyumni kamera ko‘rish maydoniga olib keling"
            textSize = 11f
            setTextColor(muted)
            gravity = Gravity.CENTER
            setPadding(0, dp(10), 0, dp(2))
        }

        root.addView(header, LinearLayout.LayoutParams(-1, -2))
        root.addView(statusCard, LinearLayout.LayoutParams(-1, -2))
        root.addView(cameraTitle, LinearLayout.LayoutParams(-1, -2))
        root.addView(cameraCard, LinearLayout.LayoutParams(-1, 0, 1.35f))
        root.addView(statsTitle, LinearLayout.LayoutParams(-1, -2))
        root.addView(statsRow, LinearLayout.LayoutParams(-1, -2))
        root.addView(footer, LinearLayout.LayoutParams(-1, -2))
        setContentView(root)
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun roundedBackground(
        color: Int,
        radiusDp: Float,
        strokeColor: Int? = null,
        strokeWidthPx: Int = 0
    ): android.graphics.drawable.GradientDrawable {
        return android.graphics.drawable.GradientDrawable().apply {
            setColor(color)
            cornerRadius = dp(radiusDp.toInt()).toFloat()
            if (strokeColor != null && strokeWidthPx > 0) {
                setStroke(strokeWidthPx, strokeColor)
            }
        }
    }

    private fun createStatTile(
        title: String,
        value: String,
        valueColor: Int,
        backgroundColor: Int
    ): Pair<LinearLayout, TextView> {
        val tile = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(5), dp(7), dp(5), dp(7))
            background = roundedBackground(backgroundColor, 15f)
        }
        val titleView = TextView(this).apply {
            text = title
            textSize = 9f
            letterSpacing = 0.08f
            setTextColor(Color.rgb(151, 166, 188))
            gravity = Gravity.CENTER
        }
        val valueView = TextView(this).apply {
            text = value
            textSize = 18f
            setTextColor(valueColor)
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            gravity = Gravity.CENTER
            setPadding(0, dp(3), 0, 0)
        }
        tile.addView(titleView, LinearLayout.LayoutParams(-1, -2))
        tile.addView(valueView, LinearLayout.LayoutParams(-1, -2))
        return tile to valueView
    }

    private fun loadDetectionModel() {
        modelExecutor.execute {
            var loadedModel: ObjectDetector? = null
            try {
                val options = ObjectDetector.ObjectDetectorOptions.builder()
                    .setScoreThreshold(0.20f)
                    .setMaxResults(30)
                    .build()

                loadedModel = ObjectDetector.createFromFileAndOptions(
                    this,
                    "efficientdet-lite2.tflite",
                    options
                )

                if (!screenIsActive || isFinishing || isDestroyed) {
                    loadedModel.close()
                    return@execute
                }

                detector = loadedModel
                loadedModel = null
                runOnUiThread {
                    if (screenIsActive && !isFinishing && !isDestroyed) {
                        status.text = if (cameraStarted) {
                            "Aniqlash faol. Kamera oldidagi buyumlarni ko‘rsating."
                        } else {
                            "Sun’iy intellekt modeli tayyor. Kamera ulanmoqda…"
                        }
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    if (screenIsActive && !isFinishing && !isDestroyed) {
                        status.text = "Sun’iy intellekt modelini yuklab bo‘lmadi. Ilovani qayta ishga tushiring."
                    }
                }
            } finally {
                loadedModel?.close()
            }
        }
    }

    private fun requestCameraAccessAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        if (!screenIsActive || cameraStarted || cameraStarting) return
        cameraStarting = true
        val requestGeneration = ++cameraGeneration
        status.text = if (detector == null) {
            "Kamera ochilmoqda. Sun’iy intellekt modeli tayyorlanmoqda…"
        } else {
            "Kamera ochilmoqda…"
        }

        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                if (requestGeneration != cameraGeneration || !screenIsActive || isFinishing || isDestroyed ||
                    !lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)
                ) {
                    if (requestGeneration == cameraGeneration) cameraStarting = false
                    return@addListener
                }

                cameraProvider = provider
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(960, 540))
                    .setTargetRotation(previewView.display?.rotation ?: android.view.Surface.ROTATION_0)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                analysis.setAnalyzer(cameraExecutor) { frame -> analyzeFrame(frame) }

                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                cameraStarted = true
                cameraStarting = false
                status.text = if (detector == null) {
                    "Kamera ishlayapti. Sun’iy intellekt modeli yuklanmoqda…"
                } else {
                    "Aniqlash faol. Kamera oldidagi buyumlarni ko‘rsating."
                }
            } catch (error: Exception) {
                if (requestGeneration == cameraGeneration) {
                    cameraStarted = false
                    cameraStarting = false
                    status.text = "Kamerani ishga tushirib bo‘lmadi. Qurilma kamerasini tekshiring."
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val frameGeneration = cameraGeneration
        var originalBitmap: Bitmap? = null
        var rotatedBitmap: Bitmap? = null
        try {
            val activeDetector = detector
            if (activeDetector == null) return

            val sourceBitmap = imageProxy.toBitmap()
            originalBitmap = sourceBitmap
            val rotation = imageProxy.imageInfo.rotationDegrees
            val orientedBitmap = if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(
                    sourceBitmap, 0, 0,
                    sourceBitmap.width, sourceBitmap.height,
                    matrix, true
                )
            } else {
                sourceBitmap
            }
            rotatedBitmap = orientedBitmap

            if (orientedBitmap !== sourceBitmap && !sourceBitmap.isRecycled) {
                sourceBitmap.recycle()
                originalBitmap = null
            }

            val rawDetections = activeDetector.detect(TensorImage.fromBitmap(orientedBitmap))
            // Kamera to'xtatilgan yoki qayta ishga tushgan eski kadrlar tracker holatini o'zgartirmasin.
            if (frameGeneration != cameraGeneration || !cameraStarted || !screenIsActive) return

            val detections = objectAgentEngine.update(rawDetections)
                .map { it.boundingBox to "${uzbekObjectName(it.label)} — ${(it.score * 100).toInt()}%" }
            val sourceWidth = orientedBitmap.width
            val sourceHeight = orientedBitmap.height

            runOnUiThread {
                if (frameGeneration == cameraGeneration && screenIsActive && cameraStarted && !isFinishing && !isDestroyed) {
                    overlay.showDetections(sourceWidth, sourceHeight, detections)
                    count.text = detections.size.toString()
                    status.text = if (detections.isEmpty()) {
                        "Hozircha narsa topilmadi. Buyumni kameraga yaqinroq tuting."
                    } else {
                        "Aniqlash faol — ${detections.size} ta narsa topildi."
                    }
                }
            }
        } catch (_: Exception) {
            runOnUiThread {
                if (screenIsActive && cameraStarted && !isFinishing && !isDestroyed) {
                    status.text = "Tasvirni tahlil qilishda xatolik yuz berdi."
                }
            }
        } finally {
            if (rotatedBitmap != null && rotatedBitmap !== originalBitmap && !rotatedBitmap.isRecycled) {
                rotatedBitmap.recycle()
            }
            if (originalBitmap != null && !originalBitmap.isRecycled) originalBitmap.recycle()
            imageProxy.close()
        }
    }

    private fun uzbekObjectName(label: String): String {
        return when (label.lowercase()) {
            "person" -> "Odam"
            "bicycle" -> "Velosiped"
            "car" -> "Yengil avtomobil"
            "motorcycle" -> "Mototsikl"
            "airplane" -> "Samolyot"
            "bus" -> "Avtobus"
            "train" -> "Poyezd"
            "truck" -> "Yuk mashinasi"
            "boat" -> "Qayiq"
            "traffic light" -> "Svetofor"
            "fire hydrant" -> "Yong‘in gidranti"
            "stop sign" -> "To‘xtash belgisi"
            "parking meter" -> "Avtoturargoh hisoblagichi"
            "bench" -> "O‘rindiq"
            "bird" -> "Qush"
            "cat" -> "Mushuk"
            "dog" -> "It"
            "horse" -> "Ot"
            "sheep" -> "Qo‘y"
            "cow" -> "Sigir"
            "elephant" -> "Fil"
            "bear" -> "Ayiq"
            "zebra" -> "Zebra"
            "giraffe" -> "Jirafa"
            "backpack" -> "Orqa xalta"
            "umbrella" -> "Soyabon"
            "handbag" -> "Qo‘l sumkasi"
            "tie" -> "Bo‘yinbog‘"
            "suitcase" -> "Chamadon"
            "frisbee" -> "Uchar likopcha"
            "skis" -> "Chang‘i"
            "snowboard" -> "Qor taxtasi"
            "sports ball" -> "Sport to‘pi"
            "kite" -> "Varrak"
            "baseball bat" -> "Beysbol tayoqchasi"
            "baseball glove" -> "Beysbol qo‘lqopi"
            "skateboard" -> "G‘ildirakli taxta"
            "surfboard" -> "To‘lqin taxtasi"
            "tennis racket" -> "Tennis raketkasi"
            "bottle" -> "Butilka"
            "wine glass" -> "Qadah"
            "cup" -> "Piyola"
            "fork" -> "Sanchqi"
            "knife" -> "Pichoq"
            "spoon" -> "Qoshiq"
            "bowl" -> "Kosa"
            "banana" -> "Banan"
            "apple" -> "Olma"
            "sandwich" -> "Non orasidagi yegulik"
            "orange" -> "Apelsin"
            "broccoli" -> "Brokkoli"
            "carrot" -> "Sabzi"
            "hot dog" -> "Kolbasali non"
            "pizza" -> "Pitsa"
            "donut" -> "Ponchik"
            "cake" -> "Tort"
            "chair" -> "Stul"
            "couch" -> "Divan"
            "potted plant" -> "Gultuvakdagi o‘simlik"
            "bed" -> "Karavot"
            "dining table" -> "Ovqatlanish stoli"
            "toilet" -> "Hojatxona"
            "tv" -> "Televizor"
            "laptop" -> "Ko‘chma kompyuter"
            "mouse" -> "Sichqoncha"
            "remote" -> "Masofadan boshqarish pulti"
            "keyboard" -> "Klaviatura"
            "cell phone" -> "Qo‘l telefoni"
            "microwave" -> "Mikroto‘lqinli pech"
            "oven" -> "Pishirish pechi"
            "toaster" -> "Non qizartirgich"
            "sink" -> "Yuvinish chig‘anog‘i"
            "refrigerator" -> "Muzlatkich"
            "book" -> "Kitob"
            "clock" -> "Soat"
            "vase" -> "Guldon"
            "scissors" -> "Qaychi"
            "teddy bear" -> "Ayiqcha o‘yinchoq"
            "hair drier" -> "Soch quritgich"
            "toothbrush" -> "Tish cho‘tkasi"
            else -> "Noma’lum narsa"
        }
    }

    private fun stopCamera() {
        cameraGeneration++
        cameraStarted = false
        cameraStarting = false
        cameraProvider?.unbindAll()
        objectAgentEngine.reset()
        if (::overlay.isInitialized) overlay.clearDetections()
        if (::count.isInitialized) count.text = "0"
    }

    override fun onResume() {
        super.onResume()
        screenIsActive = true
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    override fun onPause() {
        screenIsActive = false
        stopCamera()
        super.onPause()
    }

    override fun onDestroy() {
        screenIsActive = false
        stopCamera()
        cameraExecutor.execute {
            detector?.close()
            detector = null
        }
        cameraExecutor.shutdown()
        modelExecutor.shutdown()
        super.onDestroy()
    }
}

private class DetectionOverlay(context: android.content.Context) : View(context) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 230, 145)
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 115, 75)
        style = Paint.Style.FILL
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f
        style = Paint.Style.FILL
    }

    private var sourceWidth = 1
    private var sourceHeight = 1
    private var boxes: List<Pair<RectF, String>> = emptyList()

    fun showDetections(width: Int, height: Int, values: List<Pair<RectF, String>>) {
        sourceWidth = max(1, width)
        sourceHeight = max(1, height)
        boxes = values
        invalidate()
    }

    fun clearDetections() {
        boxes = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (boxes.isEmpty() || width == 0 || height == 0) return

        val scale = max(width.toFloat() / sourceWidth, height.toFloat() / sourceHeight)
        val dx = (width - sourceWidth * scale) / 2f
        val dy = (height - sourceHeight * scale) / 2f

        boxes.forEach { (sourceRect, label) ->
            val rect = RectF(
                dx + sourceRect.left * scale,
                dy + sourceRect.top * scale,
                dx + sourceRect.right * scale,
                dy + sourceRect.bottom * scale
            )
            rect.intersect(0f, 0f, width.toFloat(), height.toFloat())
            if (rect.width() < 2f || rect.height() < 2f) return@forEach

            canvas.drawRect(rect, boxPaint)
            val textWidth = textPaint.measureText(label)
            val left = rect.left.coerceIn(0f, (width - textWidth - 18f).coerceAtLeast(0f))
            val top = (rect.top - 42f).coerceAtLeast(0f)
            canvas.drawRoundRect(left, top, left + textWidth + 16f, top + 36f, 7f, 7f, labelBackgroundPaint)
            canvas.drawText(label, left + 8f, top + 27f, textPaint)
        }
    }
}
