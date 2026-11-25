// App nativa: abre una ventana propia con la interfaz y arranca/cierra el motor por dentro.
import Cocoa
import WebKit

let PORT = 8890

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var server: Process?

    func applicationDidFinishLaunching(_ n: Notification) {
        buildMenu()
        let cfg = WKWebViewConfiguration()
        cfg.mediaTypesRequiringUserActionForPlayback = []
        if #available(macOS 12.3, *) { cfg.preferences.isElementFullscreenEnabled = true }
        web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self
        web.uiDelegate = self
        web.setValue(false, forKey: "drawsBackground")

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1240, height: 860),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.title = "Surround BT"
        window.minSize = NSSize(width: 380, height: 560)
        window.backgroundColor = NSColor(name: nil) { a in
            a.bestMatch(from: [.darkAqua]) == .darkAqua ? .black : NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1) }
        window.contentView = web
        window.center()
        window.setFrameAutosaveName("SurroundBTWindow")
        window.makeKeyAndOrderFront(nil)
        web.loadHTMLString(loadingHTML, baseURL: nil)

        startServer()
        waitAndLoad(tries: 0)
    }

    func startServer() {
        if portOpen() { return }     // ya hay uno corriendo (otra ventana / sesión)
        guard let res = Bundle.main.resourcePath else { return }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: res + "/server/sbt-server")
        let logs = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let logURL = logs.appendingPathComponent("Surround BT.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        if let h = try? FileHandle(forWritingTo: logURL) { p.standardOutput = h; p.standardError = h }
        do { try p.run(); server = p } catch { NSLog("No pude arrancar el motor: \(error)") }
    }

    func waitAndLoad(tries: Int) {
        if portOpen() {
            web.load(URLRequest(url: URL(string: "http://127.0.0.1:\(PORT)/")!))
        } else if tries < 120 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { self.waitAndLoad(tries: tries + 1) }
        } else {
            web.loadHTMLString(errorHTML, baseURL: nil)
        }
    }

    func portOpen() -> Bool {
        let s = socket(AF_INET, SOCK_STREAM, 0)
        if s < 0 { return false }
        defer { close(s) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(UInt16(PORT).bigEndian)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        return withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(s, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0 }
        }
    }

    // Enlaces externos (descargar BlackHole, etc.) en el navegador.
    func webView(_ w: WKWebView, decidePolicyFor a: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let u = a.request.url, let h = u.host, h != "127.0.0.1", h != "localhost", u.scheme?.hasPrefix("http") == true {
            NSWorkspace.shared.open(u); decisionHandler(.cancel); return
        }
        decisionHandler(.allow)
    }

    // Diálogos de JavaScript (confirm/alert): sin esto WKWebView responde «cancelar» solo y, por ejemplo,
    // «Quitar parlante» no hacía nada.
    func webView(_ w: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let a = NSAlert()
        a.messageText = message
        a.addButton(withTitle: "Aceptar")
        a.addButton(withTitle: "Cancelar")
        a.beginSheetModal(for: window) { completionHandler($0 == .alertFirstButtonReturn) }
    }

    func webView(_ w: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let a = NSAlert()
        a.messageText = message
        a.addButton(withTitle: "Aceptar")
        a.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ n: Notification) { server?.terminate() }

    func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let app = NSMenu()
        app.addItem(withTitle: "Acerca de Surround BT", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "Ocultar Surround BT", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        app.addItem(withTitle: "Salir de Surround BT", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = app
        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edición")
        edit.addItem(withTitle: "Cortar", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copiar", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Pegar", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Seleccionar todo", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        let winItem = NSMenuItem(); main.addItem(winItem)
        let win = NSMenu(title: "Ventana")
        win.addItem(withTitle: "Minimizar", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        win.addItem(withTitle: "Recargar", action: #selector(reload), keyEquivalent: "r")
        winItem.submenu = win
        NSApp.mainMenu = main
    }
    @objc func reload() { web.reload() }

    let loadingHTML = """
    <body style="margin:0;height:100vh;display:grid;place-items:center;background:#f5f5f7;font:600 17px -apple-system;color:#6e6e73">
    Abriendo Surround BT…</body>
    """
    let errorHTML = """
    <body style="margin:0;height:100vh;display:grid;place-items:center;background:#f5f5f7;font:17px -apple-system;color:#1d1d1f;text-align:center;padding:24px">
    <div><b>No pude arrancar el motor de sonido.</b><br><br>Cierra la app y ábrela de nuevo. Si sigue, revisa el registro en<br>
    ~/Library/Logs/Surround BT.log</div></body>
    """
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.activate(ignoringOtherApps: true)
app.run()
