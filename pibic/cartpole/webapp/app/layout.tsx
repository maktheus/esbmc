import type { Metadata } from 'next';
import './globals.css';
import Nav from '@/components/Nav';

export const metadata: Metadata = {
  title: 'DQN Cart-Pole — Verificação Formal',
  description: 'Visualização e verificação formal de controlador DQN para o sistema Cart-Pole',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body className="min-h-screen bg-gray-900 text-gray-100">
        <Nav />
        <main className="container mx-auto px-4 py-6 max-w-6xl">
          {children}
        </main>
        <footer className="border-t border-gray-800 mt-12 py-4 text-center text-gray-500 text-xs">
          DQN Cart-Pole — Verificação Formal com ESBMC &nbsp;|&nbsp; PIBIC
        </footer>
      </body>
    </html>
  );
}
