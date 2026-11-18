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
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(14, 14, 14, 12)
            setBackgroundColor(Color.rgb(247, 248, 250))
        }
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(14 + systemBars.left, 14 + systemBars.top, 14 + systemBars.right, 12 + systemBars.bottom)
            insets
        }

        val heading = TextView(this).apply {
            text = "BILMASVOY"
            textSize = 21f
            setTextColor(Color.rgb(20, 30, 48))
            gravity = Gravity.CENTER
            setCompoundDrawablesWithIntrinsicBounds(R.drawable.bilmasvoy_logo, 0, 0, 0)
            compoundDrawablePadding = 12
            setPadding(0, 5, 0, 8)
        }

        status = TextView(this).apply {
            text = "Sun’iy intellekt modeli yuklanmoqda…"
            textSize = 14f
            setTextColor(Color.DKGRAY)
            gravity = Gravity.CENTER
            setPadding(4, 6, 4, 10)
        }

        val cameraFrame = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
            contentDescription = "Jonli kamera tasviri"
        }
        overlay = DetectionOverlay(this)
        cameraFrame.addView(previewView, FrameLayout.LayoutParams(-1, -1))
        cameraFrame.addView(overlay, FrameLayout.LayoutParams(-1, -1))

        count = TextView(this).apply {
            text = "Aniqlangan narsalar: 0 ta"
            textSize = 17f
            setTextColor(Color.rgb(20, 30, 48))
            gravity = Gravity.CENTER
            setPadding(0, 10, 0, 8)
        }

        val explanation = TextView(this).apply {
            text = "Kamera oldiga buyumlarni tuting. Dastur ularni suratga olmasdan, uzluksiz aniqlaydi."
            textSize = 12f
            setTextColor(Color.DKGRAY)
            gravity = Gravity.CENTER
            setPadding(8, 2, 8, 8)
        }

        root.addView(heading)
        root.addView(status)
        root.addView(cameraFrame, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(count)
        root.addView(explanation)
        setContentView(root)
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
                    count.text = "Aniqlangan narsalar: ${detections.size} ta"
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
        if (::count.isInitialized) count.text = "Aniqlangan narsalar: 0 ta"
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
