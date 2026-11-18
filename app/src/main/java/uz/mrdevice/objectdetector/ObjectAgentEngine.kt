package uz.mrdevice.objectdetector

import android.graphics.RectF
import org.tensorflow.lite.task.vision.detector.Detection
import kotlin.math.max
import kotlin.math.min

/**
 * On-device multi-agent object tracker.
 * 80 ta COCO sinfi uchun 80 ta yengil sinf-agent ishlaydi.
 * Har bir agent faqat o'z sinfiga tegishli deteksiyalarni tasdiqlaydi va kuzatadi.
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
                val match = tracks
                    .filter { it !in touched }
                    .map { track -> track to intersectionOverUnion(track.box, candidate.box) }
                    .filter { it.second >= MATCH_IOU }
                    .maxByOrNull { it.second }
                    ?.first

                if (match != null) {
                    match.box = smooth(match.box, candidate.box)
                    match.score = candidate.score
                    match.hits = min(MAX_CONFIRMATION_HITS, match.hits + 1)
                    match.misses = 0
                    match.confirmed = match.hits >= REQUIRED_HITS
                    touched += match
                } else {
                    val newTrack = Track(
                        box = RectF(candidate.box),
                        score = candidate.score
                    )
                    tracks += newTrack
                    touched += newTrack
                }
            }

            tracks.filter { it !in touched }.forEach { it.misses++ }
            tracks.removeAll { it.misses > MAX_MISSED_FRAMES }

            return tracks
                .filter { it.confirmed && it.misses <= MAX_DISPLAY_HOLD_FRAMES }
                .map { Track(RectF(it.box), it.score, it.hits, it.misses, it.confirmed) }
        }

        fun reset() = tracks.clear()

        private fun smooth(previous: RectF, current: RectF): RectF {
            // Joriy kadrga ko'proq og'irlik beriladi: ramka silliq, lekin kechikishi kam.
            val oldWeight = 0.35f
            val newWeight = 1f - oldWeight
            return RectF(
                previous.left * oldWeight + current.left * newWeight,
                previous.top * oldWeight + current.top * newWeight,
                previous.right * oldWeight + current.right * newWeight,
                previous.bottom * oldWeight + current.bottom * newWeight
            )
        }

        companion object {
            private const val MATCH_IOU = 0.12f
            private const val REQUIRED_HITS = 2
            private const val MAX_CONFIRMATION_HITS = 3
            private const val MAX_MISSED_FRAMES = 2
            private const val MAX_DISPLAY_HOLD_FRAMES = 1
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
            val agent = agents[key] ?: return@forEach
            if (category.score < thresholdFor(key)) return@forEach

            grouped.getOrPut(key) { mutableListOf() }
                .add(Candidate(RectF(detection.boundingBox), category.score))
        }

        return agents.mapNotNull { (label, agent) ->
            val accepted = agent.update(grouped[label].orEmpty())
            if (accepted.isEmpty()) null
            else accepted.map { AgentDetection(RectF(it.box), label, it.score) }
        }.flatten()
    }

    @Synchronized
    fun reset() {
        agents.values.forEach { it.reset() }
    }

    private fun normalize(value: String): String =
        value.trim().lowercase().replace('_', ' ')

    private fun thresholdFor(label: String): Float = when (label) {
        // O'xshash ko'rinishdagi mayda sinflarda yolg'on aniqlashni qattiqroq filtrlash.
        "remote", "mouse", "cell phone", "toothbrush", "hair drier", "scissors" -> 0.62f
        else -> 0.52f
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union > 0f) intersection / union else 0f
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
