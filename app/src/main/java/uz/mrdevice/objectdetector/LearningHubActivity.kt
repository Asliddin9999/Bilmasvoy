package uz.mrdevice.objectdetector

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.core.content.ContextCompat

class LearningHubActivity : ComponentActivity() {
    private lateinit var webView: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            settings.allowContentAccess = false
            settings.javaScriptCanOpenWindowsAutomatically = false
            settings.setSupportMultipleWindows(false)
            webChromeClient = WebChromeClient()
            webViewClient = WebViewClient()
            addJavascriptInterface(LearningBridge(), "MyWorldBridge")
            loadUrl("file:///android_asset/my-world-ai/index.html")
        }
        setContentView(webView)

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) webView.goBack() else finish()
            }
        })
    }

    inner class LearningBridge {
        @JavascriptInterface
        fun openCamera() {
            runOnUiThread {
                startActivity(Intent(this@LearningHubActivity, MainActivity::class.java))
            }
        }

        @JavascriptInterface
        fun askClaude(question: String) {
            runOnUiThread {
                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                clipboard.setPrimaryClip(ClipData.newPlainText("My World AI savoli", question))
                Toast.makeText(
                    this@LearningHubActivity,
                    "Savol nusxalandi. Claude ochilgach, uni joylashtiring.",
                    Toast.LENGTH_LONG
                ).show()
                val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://claude.ai/new"))
                try {
                    startActivity(intent)
                } catch (_: Exception) {
                    Toast.makeText(this@LearningHubActivity, "Brauzer topilmadi.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onDestroy() {
        webView.removeJavascriptInterface("MyWorldBridge")
        webView.destroy()
        super.onDestroy()
    }
}
