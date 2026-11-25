// Reconecta un parlante Bluetooth ya emparejado (por su dirección MAC).
// Uso: btconnect <dirección> [--reset]   (--reset: desconecta y vuelve a conectar, fuerza el perfil de audio)
import Foundation
import IOBluetooth

let args = CommandLine.arguments
guard args.count > 1, let dev = IOBluetoothDevice(addressString: args[1]) else {
    print("uso: btconnect <dirección> [--reset]"); exit(2)
}
if args.contains("--reset") && dev.isConnected() {
    dev.closeConnection()
    Thread.sleep(forTimeInterval: 2.0)
}
if dev.isConnected() { print("ya conectado"); exit(0) }
let r = dev.openConnection()
print(r == kIOReturnSuccess ? "conectado" : "no se pudo (\(r)): ¿está encendido?")
exit(r == kIOReturnSuccess ? 0 : 1)
