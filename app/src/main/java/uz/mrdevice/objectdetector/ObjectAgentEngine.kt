package uz.mrdevice.objectdetector

import android.graphics.RectF
import org.tensorflow.lite.task.vision.detector.Detection
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * On-device class-specialist tracker.
 * Bitta TFLite model natijasini 80 ta COCO sinf agenti alohida filtrlab kuzatadi.
 */
data class AgentDetection(
    val boundingBox: RectF,
    val label: String,
    val score: Float
)

class ObjectAgentEngine {
    private data class Candidate(val box: RectF, val score: Float)

    private class Track(
        var box: RectF,
        var score: Float,
        var hits: Int = 1,
        var misses: Int = 0,
        var confirmed: Boolean = false
    )

    private class ObjectAgent(
        private val confidenceThreshold: Float
    ) {
        private val tracks = mutableListOf<Track>()

        fun update(candidates: List<Candidate>): List<Track> {
            val touched = mutableSetOf<Track>()

            candidates.sortedByDescending { it.score }.forEach { candidate ->
                val match = tracks.asSequence()
                    .filter { it !in touched }
                    .map { track -> track to associationQuality(track.box, candidate.box) }
                    .filter { it.second >= MIN_ASSOCIATION_QUALITY }
                    .maxByOrNull { it.second }
                    ?.first

                if (match != null) {
                    // EMA: koordinata tebranishini yumshatadi, lekin harakatga tez moslashadi.
                    match.box = smooth(match.box, candidate.box)
                    match.score = match.score * 0.35f + candidate.score * 0.65f
                    match.hits = min(MAX_CONFIRMATION_HITS, match.hits + 1)
                    match.misses = 0
                    match.confirmed = match.hits >= REQUIRED_HITS
                    touched.add(match)
                } else {
                    val newTrack = Track(
                        box = RectF(candidate.box),
                        score = candidate.score
                    )
                    tracks.add(newTrack)
                    touched.add(newTrack)
                }
            }

            tracks.filter { it !in touched }.forEach { it.misses++ }
            tracks.removeAll { it.misses > MAX_MISSED_FRAMES }

            return tracks
                .filter { it.confirmed && it.misses <= MAX_DISPLAY_HOLD_FRAMES }
                .map { Track(RectF(it.box), it.score, it.hits, it.misses, it.confirmed) }
        }

        fun reset() {
            tracks.clear()
        }

        private fun smooth(previous: RectF, current: RectF): RectF {
            val oldWeight = 0.45f
            val newWeight = 1f - oldWeight
            return RectF(
                previous.left * oldWeight + current.left * newWeight,
                previous.top * oldWeight + current.top * newWeight,
                previous.right * oldWeight + current.right * newWeight,
                previous.bottom * oldWeight + current.bottom * newWeight
            )
        }

        /**
         * IoU alone tez harakatda trackni yo'qotishi mumkin.
         * Shu sababli IoU va markazlar orasidagi masofa birgalikda baholanadi.
         */
        private fun associationQuality(a: RectF, b: RectF): Float {
            val iou = intersectionOverUnion(a, b)
            val acx = (a.left + a.right) * 0.5f
            val acy = (a.top + a.bottom) * 0.5f
            val bcx = (b.left + b.right) * 0.5f
            val bcy = (b.top + b.bottom) * 0.5f
            val distance = sqrt((acx - bcx) * (acx - bcx) + (acy - bcy) * (acy - bcy))
            val diagonal = max(
                sqrt(a.width() * a.width() + a.height() * a.height()),
                sqrt(b.width() * b.width() + b.height() * b.height())
            )
            val allowedDistance = max(MIN_CENTER_DISTANCE_PX, diagonal * MAX_CENTER_DISTANCE_RATIO)
            val proximity = if (allowedDistance > 0f) {
                (1f - distance / allowedDistance).coerceIn(0f, 1f)
            } else 0f
            return max(iou, proximity * CENTER_ASSOCIATION_WEIGHT)
        }

        private fun intersectionOverUnion(a: RectF, b: RectF): Float {
            val left = max(a.left, b.left)
            val top = max(a.top, b.top)
            val right = min(a.right, b.right)
            val bottom = min(a.bottom, b.bottom)
            val intersection = max(0f, right - left) * max(0f, bottom - top)
            val areaA = max(0f, a.width()) * max(0f, a.height())
            val areaB = max(0f, b.width()) * max(0f, b.height())
            val union = areaA + areaB - intersection
            return if (union > 0f) intersection / union else 0f
        }

        companion object {
            private const val REQUIRED_HITS = 2
            private const val MAX_CONFIRMATION_HITS = 3
            private const val MAX_MISSED_FRAMES = 2
            private const val MAX_DISPLAY_HOLD_FRAMES = 1
            private const val MIN_ASSOCIATION_QUALITY = 0.18f
            private const val MIN_CENTER_DISTANCE_PX = 14f
            private const val MAX_CENTER_DISTANCE_RATIO = 0.35f
            private const val CENTER_ASSOCIATION_WEIGHT = 0.55f
        }
    }

    private val agents: Map<String, ObjectAgent> = COCO_LABELS.associateWith { label ->
        ObjectAgent(thresholdFor(label))
    }

    @Synchronized
    fun update(detections: List<Detection>): List<AgentDetection> {
        val grouped = mutableMapOf<String, MutableList<Candidate>>()

        detections.forEach { detection ->
            val category = detection.categories.maxByOrNull { it.score } ?: return@forEach
            val key = normalize(category.label)
            if (key !in agents || category.score < thresholdFor(key)) return@forEach

            grouped.getOrPut(key) { mutableListOf() }
                .add(Candidate(RectF(detection.boundingBox), category.score))
        }

        return agents.flatMap { (label, agent) ->
            agent.update(grouped[label].orEmpty()).map {
                AgentDetection(RectF(it.box), label, it.score)
            }
        }
    }

    @Synchronized
    fun reset() {
        agents.values.forEach { it.reset() }
    }

    private fun normalize(value: String): String =
        value.trim().lowercase().replace('_', ' ')

    private fun thresholdFor(label: String): Float = when (label) {
        "remote", "mouse", "cell phone", "toothbrush", "hair drier", "scissors" -> 0.62f
        else -> 0.52f
    }

    companion object {
        private val COCO_LABELS = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }
}
